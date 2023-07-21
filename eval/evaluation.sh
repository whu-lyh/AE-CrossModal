#!/usr/bin/env bash

# local env
# python evaluateSequences.py --dataset_root_dir "/data-lyh" \
#     --resume_path2d "/pr/AE-CrossModal/weights/checkpoint_epoch49.pth.tar" \
#     --resume_path3d "/pr/AE-CrossModal/weights/checkpoint_epoch49.ckpt" \
#     --save_path "/pr/AE-CrossModal/eval/output_lyh" \
#     --attention \
#     --debug

# remote server
python evaluateSequences.py --dataset_root_dir "/workspace/kitti360" \
    --save_path "/workspace/AE-CrossModal/eval/output_lyh" \
    --resume_path2d "/workspace/AE-CrossModal/log/checkpoints/Jul06_04_57_54_train3_val3/checkpoints/model_best.pth.tar" \
    --resume_path3d "/workspace/AE-CrossModal/log/checkpoints/Jul06_04_57_54_train3_val3/checkpoints3d/model_best.ckpt" \
    --attention
    # \
    #--debug
# zhao's weight
    #--resume_path2d "/root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.pth.tar" \
    #--resume_path3d "/root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.ckpt" \

# feature similarity
# python evaluatePairs.py --image "/pr/AE-CrossModal/eval/tmp_data/0000000111.png" \
#     --pc "/pr/AE-CrossModal/eval/tmp_data/0000000437.bin" \
#     --resume_path2d "/pr/AE-CrossModal/weights/checkpoint_epoch49.pth.tar" \
#     --resume_path3d "/pr/AE-CrossModal/weights/checkpoint_epoch49.ckpt" \
#     --save_path "/pr/AE-CrossModal/eval/output_lyh" \
#     --attention \
#     --debug

