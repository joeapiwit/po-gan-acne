# Data Directory

## Splits (included in repo)

The `splits/` directory contains stratified 5-fold cross-validation split files for both datasets. These files define which images belong to train, validation, and test sets for each fold.

Each line follows the format: `<filename> <severity_class> <lesion_count>`

- `severity_class`: 0 (mild), 1 (moderate), 2 (severe), 3 (very severe) based on Hayashi grading
- `lesion_count`: number of papules and pustules on the half-face

## Real Datasets (download separately)

### ACNE04
- 1,457 images, 4 severity classes
- Download from the original authors (Wu et al., ICCV 2019)
- Place all images in `acne04/` (flat directory, filenames like `levle0_0.jpg`, `levle1_100.jpg`, etc.)

### AcneSCU
- 276 images, 4 severity classes
- Download from the original repository (Shen et al., Sichuan University)
- Place all images in `acnescu/` (flat directory, filenames like `0.jpg`, `1.jpg`, ..., `275.jpg`)

## Synthetic Images (generated from GAN weights)

Synthetic images are not included in the repository. To reproduce:

1. Download pre-trained GAN weights from [Zenodo link TBD]
2. Place `.pkl` files in `../weights/`
3. Generate images using the generator module:

```bash
cd ../generator

# For ACNE04 (example: 150 images per class)
python generate.py --outdir=../data/acne04/fakes/class_0 --seeds=0-149 \
    --network=../weights/acne04_class0.pkl

# For AcneSCU
python generate.py --outdir=../data/acnescu/fakes/class_0 --seeds=0-149 \
    --network=../weights/acnescu_class0.pkl
```

The optimizer (`optimize_ldl.py`) handles synthetic image injection automatically based on the c-vector (per-class count policy) it is evaluating.

## Directory Structure After Setup

```
data/
├── splits/
│   ├── acne04/          # 15 split files (included in repo)
│   └── acnescu/         # 15 split files (included in repo)
├── acne04/              # Real ACNE04 images (download)
├── acnescu/             # Real AcneSCU images (download)
└── README.md            # This file
```
