
import os
import sys
import argparse
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
sys.path.append("..")
from mycode.NetVLAD.netvlad import get_model_netvlad
from crossmodal.models.models_generic import get_backend, get_model
import model3d.PointNetVlad as PNV
from sphereModel.sphereresnet import sphere_resnet18
from mycode.loading_pointclouds import load_pc_file
from crossmodal.tools.datasets import input_transform

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def fetch_feature_img(query_img, model2d):
    input_data = (query_img.unsqueeze(0)).to("cuda")
    model2d.eval()
    with torch.no_grad():
        image_encoding = model2d.encoder(input_data)
        print("image_encoding.shape:\t", image_encoding.shape) # should be torch.Size([1, 512, 16, 32])
        #print('image_encoding')
        #print(image_encoding)
        vlad_encoding = model2d.pool(image_encoding)
        print("vlad_encoding.shape:\t", vlad_encoding.shape) # should be torch.Size([1, 256])
        #print('vlad_encoding')
        #print(vlad_encoding)
        return vlad_encoding
        

def fetch_feature_pc(query_pc, model3d):
    query3d = torch.tensor(query_pc)
    input_data = query3d
    input_data = input_data.view((-1, 1, 4096, 3))
    print("input.size:\t", input_data.shape)  # should be torch.Size([1, 1, 4096, 3])
    model3d.eval()
    with torch.no_grad():
        input_data = input_data.to("cuda")
        pc_encoding = model3d.point_net(input_data)
        print("pc_encoding.shape:\t", pc_encoding.shape) # should be torch.Size([1, 1024, 4096, 1])
        # print('pc_encoding')
        # print(pc_encoding)
        vlad_encoding = model3d.net_vlad(pc_encoding)
        print("vlad_encoding.shape", vlad_encoding.shape) # should be torch.Size([1, 256])
        #print('vlad_encoding')
        #print(vlad_encoding)
        return vlad_encoding


def pair_check(network, image_file, pc_file, save_path, resume_path2d, resume_path3d, attention=True, patchnv=False, debug=False):
    # load data
    tf_transform = input_transform(False)
    query_img = tf_transform(Image.open(image_file))
    query_pc = load_pc_file(pc_file)
    # mandatory in cuda
    device = torch.device("cuda")
    config = {'network': 'spherical', 'num_clusters': 64, 'pooling': 'netvlad', 'vladv2': False}
    config['train'] = {'batchsize': 10}   
    # model construction  
    if network == 'spherical':
        encoder = sphere_resnet18(pretrained=True)
        encoder_dim = 512
        # encoder_dim, encoder = get_spherical_cnn(network='original')  # TODO: freeze pretrained
    elif network == 'resnet':
        encoder_dim, encoder = get_backend(net='resnet')  # resnet
    elif network == 'vgg':
        encoder_dim, encoder = get_backend(net='vgg', pre=True)  # resnet
    else:
        raise ValueError('Unknown cnn network')   
    if patchnv:
        model2d = get_model(encoder, encoder_dim, config, append_pca_layer=False)
    else:
        model2d = get_model_netvlad(encoder, encoder_dim, config, attention)
    if attention:
        # attention mechanism by AE-Spherical
        model3d = PNV.PointNetVlad_attention(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    else:
        model3d = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    # load pretrained weights
    checkpoint = torch.load(resume_path2d, map_location=lambda storage, loc: storage)
    checkpoint3d = torch.load(resume_path3d, map_location=lambda storage, loc: storage)
    model2d.load_state_dict(checkpoint['state_dict'])
    model3d.load_state_dict(checkpoint3d['state_dict'])
    model2d = model2d.to(device)
    model3d = model3d.to(device)
    # get query feature
    print("query_img.shape:\t", query_img.shape) # should be torch.Size([3, 512, 1024])
    feat_img = fetch_feature_img(query_img, model2d)
    print("query_pc.shape:\t",query_pc.shape) # should be (4093, 3)
    feat_pc = fetch_feature_pc(query_pc, model3d)
    # print similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_similarity = cos(feat_img, feat_pc)
    print("cos_similarity:\t", cos_similarity)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CrossModal-evaluation-pair')
    parser.add_argument('--image', type=str, default='', required=True, 
                        help='Single image')
    parser.add_argument('--pc', type=str, default='', required=True, 
                        help='Single submap')
    parser.add_argument('--save_path', type=str, default='/log/checkpoints', required=True, 
                        help='Path to save checkpoints to')
    parser.add_argument('--network', type=str, default='spherical', 
                        help='2D CNN network, e.g. vgg, resnet, spherical')
    parser.add_argument('--resume_path2d', type=str, default='', required=True, 
                        help='Full path and name (with extension .pth.tar) to load checkpoint from, for 2d resuming training.') 
    parser.add_argument('--resume_path3d', type=str, default='', required=True, 
                        help='Full path and name (with extension .ckpt) to load checkpoint from, for 3d resuming training.')
    parser.add_argument('--attention', action='store_true', 
                        help='If true, add SE attention to backbone.')
    parser.add_argument('--patchnv', action='store_true', 
                        help='If true, using PatchNetVLAD as backbone.')
    parser.add_argument('--debug', action='store_true', 
                        help='If true, print additional informations.')
    opt = parser.parse_args()
    print(opt)
    opt.save_path = os.path.join(opt.save_path, datetime.now().strftime('Results_%b%d_%H_%M_%S'))
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if opt.attention:
        attention = True
    patchnv = False
    if opt.patchnv:
        patchnv = True
    if opt.debug:
        debug = True
    pair_check(network=opt.network, image_file=opt.image, pc_file=opt.pc, save_path=opt.save_path, 
             resume_path2d=opt.resume_path2d, resume_path3d=opt.resume_path3d, 
             attention=attention, patchnv=patchnv, debug=debug)
    print("Evaluation Done!")