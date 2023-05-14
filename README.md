# Basic version code for "Attention-Enhanced Cross-modal Localization Between Spherical Images and Point Clouds“

## Data
The dataset is built based on [KITTI360](https://www.cvlibs.net/datasets/kitti-360/)

Stitch dual-fisheye into spherical image by [FFmpeg](https://ffmpeg.org/)

```
# 
ffmpeg -y -i $file -vf v360=dfisheye:e:yaw=-90:ih_fov=187.8:iv_fov=185 -c:v libx265 -b:v 40000k -bufsize 5000k -preset ultrafast -c:a copy out.mp4
# -i 输入视频，-f 指定输出格式为图像，图像命名规则是basename+$file
ffmpeg -i out.mp4 -f image2 ./$(basename $file .png).png
```

For global Lidar map making and sub-maps division, refer to [this](https://github.com/Zhaozhpe/kitti360-map-python)

# some tricks
both fisheye image and point cloud are under GPS/IMU coordinate, but i believe the unify local coordinate is ok(without verified)
the final used image number of sum of query and database might smaller than the total image number
due to the filter of the too close image by KITTI360 author.

## How to use
the ML environment is based on `PyTorch 1.7.0`

Before running `mytrain.py`, specific 2D dataset directory path in `mytrain.py` or when input the command.

```--dataset_root_dir refers to the root directory of the 2D dataset```

Specify the 3D dataset directory in `./mycode/msls.py`

```path_to_3d = " "```

Train the network

```python mytrain.py```

