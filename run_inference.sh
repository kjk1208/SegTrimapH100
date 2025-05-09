#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0  

USER_ID=$(whoami)

#User input
WEIGHT_PATH="./output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/020.pth"
#User input

if [ "$USER_ID" == "kjk" ]; then
    echo "[INFO] Running on A6000 서버 (사용자: kjk)"
    DATA_PATH="./datasets/3.AIM-500"
elif [ "$USER_ID" == "work" ]; then
    echo "[INFO] Running on H100 서버 (사용자: work)"    
    DATA_PATH="./datasets/AIM-500"
else
    echo "[ERROR] Unknown user: $USER_ID"
    exit 1
fi

python inference.py \
  --ckpt_path "$WEIGHT_PATH" \
  --data_root "$DATA_PATH" \
  --save_dir ./inference/ \
  --infer_img_size 448 \
  --batch_size 4