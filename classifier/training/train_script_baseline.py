#!/usr/bin/env python3
# train_script_baseline.py — pure-classifier trainer with LDL-compatible logs

import os, csv, argparse, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import dataset_processing
from utils.report import (report_precision_se_sp_yi,
                          report_mae_mse,
                          report_precision_se_sp_yi_standard_format)
from utils.utils import Logger, AverageMeter, time_to_str
from model_baselines import build_baseline
from transforms.affine_transforms import RandomRotate


warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.manual_seed(42); np.random.seed(42)

# ─────────────────────────────────────────────────────────────
BATCH_SIZE, BATCH_SIZE_TEST = 32, 20
LR, NUM_WORKERS            = 0.001, 12
LR_STEPS                   = [30, 60, 90, 120]
MAX_NO_IMPROVE_EPOCHS      = 10
WEIGHT_SAVE                = False
# ─────────────────────────────────────────────────────────────

def evaluation_metrics(model, loader, tag, loss_func, fold, epoch):
    """return (metrics_df, record_df) for one split."""
    with torch.no_grad():
        v_loss = 0.0
        y_true=y_pred=y_pred_m=l_true=l_pred=np.array([])
        model.eval(); records=[]

        for step,(x,y,l,paths) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()
            cls, cou, c2c = model(x, None)              # adapter returns probs
            # CE in your LDL eval uses c2c directly (probs). Keep parity:
            loss        = loss_func(c2c, y)
            v_loss     += loss.item()
            _,pm        = torch.max(cls+c2c,1)
            _,p         = torch.max(cls,1)
            # fake "lesion" prediction used only for MAE/MSE format
            _,pl        = torch.max(cou,1); pl=(pl+1)

            y_true = np.hstack((y_true, y.cpu()))
            y_pred = np.hstack((y_pred, p.cpu()))
            y_pred_m=np.hstack((y_pred_m, pm.cpu()))
            l_true = np.hstack((l_true, l.numpy()))
            l_pred = np.hstack((l_pred, pl.cpu()))

            for idx,(pr,path) in enumerate(zip(p.cpu(),paths)):
                records.append({"epoch":epoch,"step":tag,"data_index":step*loader.batch_size+idx,
                                "image_path":path,"gt":int(y.cpu()[idx]),"pred":int(pr)})

        v_loss /= max(1, len(loader))
        _,_,rep      = report_precision_se_sp_yi(y_pred,y_true)
        _,_,rep_m    = report_precision_se_sp_yi(y_pred_m,y_true)
        _,MAE,MSE,mr = report_mae_mse(l_true,l_pred,y_true)

        metrics_df = report_precision_se_sp_yi_standard_format(
                        y_pred, y_true, v_loss, fold, epoch, tag, 'baseline')
        return metrics_df, records

def calculate_log_metrics(df, epoch):
    df = df[(df["epoch"]==epoch) & (df["status"]=="val")]
    return (df["precision"].mean(), df["recall"].mean(),
            df["f1"].mean()       , df["specificity"].mean(),
            df["loss"].mean())

def run_full_baseline_experiment(*, data_root:str, experiment_root:str,
                                 model_key:str):
    data_root, experiment_root = Path(data_root), Path(experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    log = Logger(); LOGFILE=experiment_root/f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    log.open(str(LOGFILE), "a")

    folds = sorted([d.name for d in data_root.iterdir() if d.is_dir()]) or list("01234")
    all_metrics=[]

    # keep EXACT same normalization as LDL
    norm = transforms.Normalize([0.45815152,0.361242,0.29348266],
                                [0.2814769 ,0.226306 ,0.20132513])

    for fold in folds:
        # ───────── directories ────────────────────────────────────────────
        fold_root = experiment_root/"ldl"/fold
        for sub in ("weights","logs","csv"): (fold_root/sub).mkdir(parents=True, exist_ok=True)
        csv_file  = fold_root/"csv"/f"logcsv_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        csv.writer(open(csv_file,"w",newline='')).writerow(
            ["status","epoch","loss_cls","loss_cou","loss_cls_cou","loss",
             "test_loss","test_acc","pre_se_sp_yi_report",
             "pre_se_sp_yi_report_m","mae_mse_report"])

        # ───────── datasets & loaders ─────────────────────────────────────
        TR = data_root/f"NNEW_train_{fold}.txt"
        VA = data_root/f"NNEW_val_{fold}.txt"
        TE = data_root/f"NNEW_test_{fold}.txt"

        def ds(txt, aug): return dataset_processing.DatasetProcessingWithImagepath(
                                data_root/fold, str(txt), transform=aug)

        ds_train = ds(TR, transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotate(20),
            norm
        ]))
        ds_val   = ds(VA, transforms.Compose([
                        transforms.Resize((224,224)), transforms.ToTensor(), norm]))
        ds_test  = ds(TE, transforms.Compose([
                        transforms.Resize((224,224)), transforms.ToTensor(), norm]))

        def loader(d,b,sh=False): return DataLoader(d,b,shuffle=sh,
                                                    num_workers=NUM_WORKERS,pin_memory=True)
        L_tr,L_va,L_te = loader(ds_train,BATCH_SIZE,True), loader(ds_val,BATCH_SIZE_TEST), loader(ds_test,BATCH_SIZE_TEST)

        # ───────── model / optimiser ──────────────────────────────────────
        net = build_baseline(model_key).cuda()
        cudnn.benchmark=True
        opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        ce  = nn.CrossEntropyLoss().cuda()

        best_loss, no_imp, start = np.inf, 0, time.time()

        for ep in range(LR_STEPS[-1]):
            if ep in LR_STEPS:
                for g in opt.param_groups: g["lr"] *= 0.5

            # ─── training (pure CE on logits via adapter) ───────────────
            net.train(); m_cls=m_cou=m_c2c=m_loss=AverageMeter()
            for bx,by,bl,_ in L_tr:
                bx=bx.cuda(); by=by.cuda()
                cls,cou,c2c = net(bx,None)   # adapter returns probabilities
                # For CE we need logits; take log-softmax inverse by using log probs:
                # Your eval uses CE on probs; to keep parity, we also use CE on probs:
                loss = ce(c2c, by)

                opt.zero_grad(); loss.backward(); opt.step()
                m_cls.update(loss.item(),bx.size(0))  # treat as "loss_cls"
                m_cou.update(0.0,bx.size(0))
                m_c2c.update(0.0,bx.size(0))
                m_loss.update(loss.item(),bx.size(0))

            log.write(f"{model_key} train {ep:3d} | {m_cls.avg:.3f} {m_cou.avg:.3f} "
                      f"{m_c2c.avg:.3f} {m_loss.avg:.3f} | "
                      f"{time_to_str(time.time()-start,'min')}\n")
            csv.writer(open(csv_file,"a",newline='')).writerow(
                ["train",ep,m_cls.avg,m_cou.avg,m_c2c.avg,m_loss.avg,None,None,None,None,None])

            # ─── evaluation (train/val/test) ─────────────────────────────
            for tag,ld in (("train",L_tr),("val",L_va),("test",L_te)):
                met,rec = evaluation_metrics(net,ld,tag,ce,fold,ep)
                pd.DataFrame(met).to_csv(fold_root/"logs"/f"epoch_{ep}_{tag}.csv", index=False)
                pd.DataFrame(rec).to_csv(fold_root/"logs"/f"epoch_{ep}_{tag}_detail.csv",index=False)
                all_metrics += met

            # ─── early stop (val loss) ──────────────────────────────────
            _,_,_,_,val_loss = calculate_log_metrics(pd.DataFrame(all_metrics), ep)
            if val_loss < best_loss: best_loss = val_loss; no_imp=0
            else:
                no_imp += 1
                if no_imp >= MAX_NO_IMPROVE_EPOCHS: break

        # save weights per fold (optional)
        if WEIGHT_SAVE:
            wpath = (fold_root/"weights"/f"{model_key}_best.pth")
            torch.save(net.state_dict(), str(wpath))

    # finish all folds
    pd.DataFrame(all_metrics).to_csv(experiment_root/"eval_ldl.csv", index=False)
    log.write("✓ baseline experiment finished.\n")


# ───────── CLI ─────────
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data" , required=True, help="dataset root that contains fold subdirs and NNEW_* files")
    ap.add_argument("--out"  , required=True, help="experiment run root (will contain ldl/<fold>/... and eval_ldl.csv)")
    ap.add_argument("--model", required=True, choices=[
        "effnet_l2","effnet_b7","resnext101_32x8d","convnext_xl","vit_l16","beit_l16"
    ])
    ap.add_argument("--save_weights", action="store_true")
    args = ap.parse_args()

    WEIGHT_SAVE = args.save_weights
    run_full_baseline_experiment(
        data_root=args.data,
        experiment_root=args.out,
        model_key=args.model,
    )
