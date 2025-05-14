#!/bin/bash

USER_ID=$(whoami)

# User input
CONFIG="./models/ddp/trimap_huge448_CEloss_noposembed_ddp.py"
# User input

if [ "$USER_ID" == "kjk" ]; then
    echo "[INFO] Running on A6000 서버 (사용자: kjk)"
    NGPU=4
    BATCH=32
elif [ "$USER_ID" == "work" ]; then
    echo "[INFO] Running on H100 서버 (사용자: work)"
    NGPU=2
    BATCH=29
else
    echo "[ERROR] Unknown user: $USER_ID"
    exit 1
fi

# DDP 실행
#export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=$NGPU train.py $CONFIG \
    --batch-size $BATCH \
    --ngpus $NGPU \
    --upsample 'x4'