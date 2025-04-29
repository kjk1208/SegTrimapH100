python inference.py \
  --ckpt_path ./output/loss/composition_p3m10k_am2k_trimap_vit_huge448_focalloss_dtloss/001/checkpoints/020.pth \
  --data_root ./datasets/3.AIM-500 \
  --save_dir ./inference/ \
  --infer_img_size 448 \
  --batch_size 4