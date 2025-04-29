#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0  

WEIGHT_PATH="/home/kjk/matting/SegTrimap/output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/last_checkpoint.pth"
COMPOSITION_PATH="./datasets/Seg2TrimapDataset/Composition-1k-testset"
P3M500_PATH="./datasets/2.P3M-10k/P3M-10k/validation/P3M-500-NP"
AIM500_PATH="./datasets/Seg2TrimapDataset/AIM-500"
AM200_PATH="./datasets/Seg2TrimapDataset/AM-200"

LOG_DIR="./evaluation/eval_logs"

python evaluation/eval.py \
    --weight_path "$WEIGHT_PATH" \
    --device "cpu" \
    --batch_size 20 \
    --log_dir "$LOG_DIR" \
    --composition_path "$COMPOSITION_PATH" \
    --p3m500_path "$P3M500_PATH" \
    --aim500_path "$AIM500_PATH" \
    --am200_path "$AM200_PATH"



# MODEL_PATH=./weights/simpleclick_models/cocolvis_vit_base.pth

# python scripts/evaluate_model.py NoBRS \
# --gpu=0 \
# --checkpoint=${MODEL_PATH} \
# --eval-mode=cvpr \
# --datasets=GrabCut