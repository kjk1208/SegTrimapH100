#!/bin/bash

USER_ID=$(whoami)

#User input
CONFIG="./models/loss/trimap_huge448_CE_loss_noposembed.py"
#User input

if [ "$USER_ID" == "kjk" ]; then
    echo "[INFO] Running on A6000 서버 (사용자: kjk)"
    NGPU=4
    BATCH=32    
elif [ "$USER_ID" == "work" ]; then
    echo "[INFO] Running on H100 서버 (사용자: work)"
    NGPU=2
    BATCH=28    
else
    echo "[ERROR] Unknown user: $USER_ID"
    exit 1
fi

python train.py $CONFIG \
    --batch-size=$BATCH \
    --ngpus=$NGPU \
    --upsample='x4'

# python train.py ./models/iter_mask/plainvit_huge448_cocolvis_itermask.py \
# --batch-size=32 \
# --ngpus=4

# python train.py ./models/iter_mask/trimap_huge448.py \
# --batch-size=32 \
# --ngpus=4 \
# --upsample='x4'

# python train.py ./models/loss/trimap_huge448_CE_loss.py \
# --batch-size=28 \
# --ngpus=2 \
# --upsample='x4'