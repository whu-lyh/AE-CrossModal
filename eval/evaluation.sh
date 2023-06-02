#!/usr/bin/env bash

# local env
# python evaluateSequences.py --dataset_root_dir "/data-lyh" \
#     --resume_path2d "/pr/AE-CrossModal/weights/checkpoint_epoch49.pth.tar" \
#     --resume_path3d "/pr/AE-CrossModal/weights/checkpoint_epoch49.ckpt" \
#     --save_path "/pr/AE-CrossModal/eval/output_lyh" \
#     --attention \
#     --debug

# remote server
# python evaluateSequences.py --dataset_root_dir "/root/public/data/Kitti/kitti360" \
#     --resume_path2d "/root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.pth.tar" \
#     --resume_path3d "/root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.ckpt" \
#     --attention \
#     --debug

# feature similarity
python evaluatePairs.py --image "/pr/AE-CrossModal/eval/tmp_data/0000000111.png" \
    --pc "/pr/AE-CrossModal/eval/tmp_data/0000000437.bin" \
    --resume_path2d "/pr/AE-CrossModal/weights/checkpoint_epoch49.pth.tar" \
    --resume_path3d "/pr/AE-CrossModal/weights/checkpoint_epoch49.ckpt" \
    --save_path "/pr/AE-CrossModal/eval/output_lyh" \
    --attention \
    --debug