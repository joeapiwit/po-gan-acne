# Pre-trained Weights

This directory stores GAN checkpoint files (`.pkl`) used to generate synthetic acne images.

Weights are not included in the Git repository due to file size (~336 MB each). Download from:

- **[Google Drive](https://drive.google.com/open?id=1s4sMdC9I4yxV0hLu8gtt2Z-xnK54arUC)**

## Available Checkpoints

Each dataset has one StyleGAN2-ADA checkpoint per cross-validation fold, fine-tuned on that fold's training split (512x512 resolution, resumed from FFHQ pre-trained weights).

### ACNE04

| File | Fold | Snapshot (kimg) |
|------|:----:|:---------------:|
| `acne04/acne04_fold0.pkl` | 0 | 5800 |
| `acne04/acne04_fold1.pkl` | 1 | 7200 |
| `acne04/acne04_fold2.pkl` | 2 | 12600 |

### AcneSCU

| File | Fold | Snapshot (kimg) |
|------|:----:|:---------------:|
| `acnescu/acnescu_fold0.pkl` | 0 | 5400 |
| `acnescu/acnescu_fold1.pkl` | 1 | 4800 |
| `acnescu/acnescu_fold2.pkl` | 2 | 3600 |
| `acnescu/acnescu_fold3.pkl` | 3 | 4000 |
| `acnescu/acnescu_fold4.pkl` | 4 | 5800 |

## Usage

```bash
cd generator
python generate.py --outdir=output_dir --seeds=0-149 --network=../weights/acnescu/acnescu_fold0.pkl
```

## Training from Scratch

To fine-tune StyleGAN2-ADA on your own data:

```bash
cd generator

# 1. Prepare dataset (convert images to StyleGAN2 format)
python dataset_tool.py --source=/path/to/images --dest=/path/to/dataset.zip

# 2. Fine-tune from FFHQ pre-trained weights
python train.py --outdir=training-runs --data=/path/to/dataset.zip \
    --gpus=1 --cfg=auto --resume=ffhq256 --snap=10
```

See `generator/README.md` for full StyleGAN2-ADA documentation.
