#!/bin/bash

for fold in 0 1 2 3 4; do
  echo "Processing fold ${fold} with size 512x512..."
  python dataset_tool.py \
    --source /mnt/wd-ssd-4tb/acne_classifiers/datasets/image_restructured/${fold}/train/all \
    --dest   /mnt/wd-ssd-4tb/acne_classifiers/datasets/image_restructured/${fold}/stylegan2_format/dataset512x512.zip \
    --width 512 \
    --height 512
done
