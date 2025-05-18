#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0  

USER_ID=$(whoami)

#User input
WEIGHT_PATH="/home/work/SegTrimap/output/ddp/new_augmentation_trimap_vit_huge448_CE_loss_Focal_loss_noposembed_ddp/000/checkpoints/054.pth"
#User input

LOG_DIR="./evaluation/eval_logs"

if [ "$USER_ID" == "kjk" ]; then
    echo "[INFO] Running on A6000 서버 (사용자: kjk)"
    COMPOSITION_PATH="./datasets/Seg2TrimapDataset/Composition-1k-testset"
    P3M500_PATH="./datasets/2.P3M-10k/P3M-10k/validation/P3M-500-NP"
    AIM500_PATH="./datasets/Seg2TrimapDataset/AIM-500"
    AM200_PATH="./datasets/Seg2TrimapDataset/AM-200"  
elif [ "$USER_ID" == "work" ]; then
    echo "[INFO] Running on H100 서버 (사용자: work)"    
    COMPOSITION_PATH="./datasets/Composition-1k-testset"
    P3M500_PATH="./datasets/2.P3M-10k/P3M-10k/validation/P3M-500-NP"
    AIM500_PATH="./datasets/AIM-500"
    AM200_PATH="./datasets/AM-200"  
else
    echo "[ERROR] Unknown user: $USER_ID"
    exit 1
fi

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