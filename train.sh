#!/usr/bin/env bash
# with h5 cache
# python mytrain.py --thread 8 \
#         --network spherical \
#         --id spherical \
#         --attention \
#         --pretrained_cnn_network \
#         --dataset_root_dir /root/public/data/Kitti/kitti360 \
#         --cache_path /root/lyh/data_lyh/kitti360/cache \
#         --save_path /root/lyh/code/AE-CrossModal/log/checkpoints \
#         --cluster_path /root/lyh/data_lyh/kitti360/cache/centroids/spherical_20m_KITTI360_64_desc_cen.hdf5

# 1080Ti
#python mytrain.py --thread 0 \
#        --network spherical \
#        --id spherical \
#        --dataset_root_dir /data-lyh \
#        --cache_path /data-lyh/cache \
#        --cluster_path /data-lyh/cache/centroids/spherical_20m_KITTI360_64_desc_cen.hdf5

# without h5 cache
# due to the performance of the original model is the best, no additional model_best is saved
python mytrain.py --thread 7 \
     --network spherical \
     --attention \
     --id spherical_new_from_scratch \
     --start_epoch 0 \
     --pretrained_cnn_network \
     --cache_path /root/lyh/data_lyh/kitti360/cache \
     --dataset_root_dir /root/lyh/data_lyh/kitti360 \
     --save_path /root/lyh/code/AE-CrossModal/log/checkpoints \
     --cluster_path /root/lyh/data_lyh/kitti360/cache/centroids/spherical_20m_KITTI360_64_desc_cen.hdf5

# python mytrain.py --thread 8 \
#         --network spherical \
#         --attention \
#         --id spherical_new \
#         --start_epoch 0 \
#         --pretrained_cnn_network \
#         --cache_path /root/lyh/data_lyh/kitti360/cache \
#         --dataset_root_dir /root/public/data/Kitti/kitti360 \
#         --save_path /root/lyh/code/AE-CrossModal/log/checkpoints \
#         --resume_path2d /root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.pth.tar \
#         --resume_path3d /root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.ckpt

# weights from zhipeng
    # --resume_path2d /root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.pth.tar \
    # --resume_path3d /root/lyh/code/AE-CrossModal/weights/checkpoint_epoch49.ckpt


# for resuming train the cluster file is unnecessary
#--cluster_path /root/lyh/data_lyh/kitti360/cache/centroids/spherical_20m_KITTI360_64_desc_cen.hdf5 \


#python  -m torch.distributed.launch --nproc_per_node=2  --master_port 29501 main.py
