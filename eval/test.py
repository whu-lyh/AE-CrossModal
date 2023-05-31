import os
import numpy as np
import open3d as o3

filename = '/home/zhipengz/data/kitti360_pc/s00/pc/0000000001.bin'
pc = np.fromfile(os.path.join('', filename), dtype=np.float32) # roc edit 64->32
pc = np.reshape(pc, [-1, 4])
print("len(pc):\t",pc.shape)
# retain only 4096 point clouds
n = np.random.choice(len(pc), 4096, replace=False)
pc = pc[n]
# split xyz and intensity
pc=pc[:, :3]
local_intensity=[]
pcd_nosam = o3.geometry.PointCloud()
pcd_nosam.points = o3.utility.Vector3dVector(pc)
pcd_nosam.colors = o3.utility.Vector3dVector(local_intensity)
o3.io.write_point_cloud('./00.pcd', pcd_nosam)