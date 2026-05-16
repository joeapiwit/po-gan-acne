import pandas as pd
import os
import torch

def append_log(df, fold, epoch, model_name, num_classes, all_step_eval_metrics):
    for cls in range(num_classes):
        for stepname, stepname_eval_metrics in all_step_eval_metrics.items():
            df.append({
                'fold': fold,
                'epoch': epoch,
                'status': stepname,
                'model_name': model_name,
                'loss': stepname_eval_metrics['loss'],
                'precision': stepname_eval_metrics['precision'][cls],
                'recall': stepname_eval_metrics['recall'][cls],
                'f1': stepname_eval_metrics['f1'][cls],
                'specificity': stepname_eval_metrics['specificity'][cls],
                'class': cls
            })

    return df

def calculate_log_metrics(df, epoch):
    # filter df by epoch, st atus=val
    df = df[(df['epoch'] == epoch) & (df['status'] == 'val')]
    # calculate precision, recall, f1, specificity all classes
    precision = df['precision'].mean()
    recall = df['recall'].mean()
    f1 = df['f1'].mean()
    specificity = df['specificity'].mean()
    loss = df['loss'].mean()
    return precision, recall, f1, specificity, loss


def update_best_record_weight(model, weight_path, metric, epoch):
    # delete all previous weights. The name starts with 'best_{metric}_val_epoch'
    for file in os.listdir(weight_path):
        if file.startswith('best_{}_val_epoch'.format(metric)):
            os.remove(os.path.join(weight_path, file))

    # save the best weight
    weight_name = 'best_{}_val_epoch_{}.pth'.format(metric, epoch)
    torch.save(model.state_dict(), os.path.join(weight_path, weight_name))