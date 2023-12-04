#!/usr/bin/env python
import os
import sys
from datetime import datetime
sys.path.append("..")
import argparse
import torch
import torch.nn as nn
from mycode.detail_val import val
from mycode.msls import MSLS
from mycode.NetVLAD.netvlad import get_model_netvlad
from crossmodal.tools.datasets import input_transform
from crossmodal.models.models_generic import get_backend, get_model
import model3d.PointNetVlad as PNV
from sphereModel.sphereresnet import sphere_resnet18

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate(network, dataset_root_dir, save_path, resume_path2d, resume_path3d, attention=True, patchnv=False, debug=False):
    # mandatory in cuda
    device = torch.device("cuda")
    config = {'network': 'spherical', 'num_clusters': 64, 'pooling': 'netvlad', 'vladv2': False}
    # for evaluation, batch size could be 80 at 24GB GPU
    config['train'] = {'cachebatchsize': 40, 'batchsize': 40}
    config['global_params'] = {'threads': 8}
    if not os.path.exists(opt.resume_path2d) or not os.path.exists(opt.resume_path3d):
        print("Dummy prediction test")
        validation_dataset = MSLS(dataset_root_dir, mode='val', transform=input_transform(train=False), batch_size=config['train']['cachebatchsize'], threads=8, posDistThr=20)
        recalls = val(validation_dataset, model2d=None, model3d=None, batch_size=config['train']['batchsize'], device=device, threads=config['global_params']['threads'], result_path=save_path, pbar_position=1)
        return
    # model construction  
    if network == 'spherical':
        encoder = sphere_resnet18(pretrained=False)
        encoder_dim = 512
        # encoder_dim, encoder = get_spherical_cnn(network='original')  # TODO: freeze pretrained
    elif network == 'resnet':
        encoder_dim, encoder = get_backend(net='resnet')  # resnet
    elif network == 'vgg':
        encoder_dim, encoder = get_backend(net='vgg', pre=False)  # resnet
    else:
        raise ValueError('Unknown cnn network')   
    if patchnv:
        model2d = get_model(encoder, encoder_dim, config, append_pca_layer=False)
    else:
        model2d = get_model_netvlad(encoder, encoder_dim, config, attention=attention)
    if attention:
        # attention mechanism by AE-Spherical
        model3d = PNV.PointNetVlad_attention(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    else:
        model3d = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    # load pretrained weights
    checkpoint = torch.load(resume_path2d, map_location=lambda storage, loc: storage)
    checkpoint3d = torch.load(resume_path3d, map_location=lambda storage, loc: storage)
    if debug:
        print("chepoint2d")
        print(checkpoint.keys())
        for key in checkpoint.keys():
            print(key)
            #if key is not 'state_dict':
                #print(checkpoint[key])
        print("chepoint3d")
        print(checkpoint3d.keys())
        for key in checkpoint3d.keys():
            print(key)
            # block the parameters that are too large for visualization
            #if key is not 'state_dict':
                #print(checkpoint3d[key])
        # for name, param in model2d.named_parameters():
        #     print(name, '      ', param.size())
        # for name, param in model3d.named_parameters():
        #     print(name, '      ', param.size())  
    if torch.cuda.device_count() > 1:
        model2d = nn.DataParallel(model2d).cuda()
    model2d.load_state_dict(checkpoint['state_dict'])
    if debug:
        print("pre_model2d")
        print(model2d)
    if torch.cuda.device_count() > 1:
        model3d = nn.DataParallel(model3d).cuda()
    model3d.load_state_dict(checkpoint3d['state_dict'])
    if debug:
        print("pre_model3d")
        print(model3d) 
    model2d = model2d.to(device)
    model3d = model3d.to(device)
    # the batch_size, thread and other settings are designed for hardest sampling mining
    validation_dataset = MSLS(dataset_root_dir, mode='val', transform=input_transform(train=False), 
                              batch_size=config['train']['cachebatchsize'], posDistThr=20)
    val(validation_dataset, model2d, model3d, batch_size=config['train']['batchsize'], threads=config['global_params']['threads'], 
        result_path=save_path, device=device, faiss_gpu=True if device=="cuda" else False, save_fig=False, pbar_position=1, debug=debug)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CrossModal-evaluation-sequence')
    parser.add_argument('--dataset_root_dir', type=str, default='/data-lyh/KITTI360', required=True, 
                        help='Root directory of dataset')
    parser.add_argument('--save_path', type=str, default='/log/checkpoints', required=True, 
                        help='Path to save checkpoints')
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
    opt.save_path = os.path.join(opt.save_path, datetime.now().strftime('EvaluationResults_%b%d_%H_%M_%S'))
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    attention = False
    if opt.attention:
        attention = True
    patchnv = False
    if opt.patchnv:
        patchnv = True
    debug = False
    if opt.debug:
        debug = True
    evaluate(network=opt.network, dataset_root_dir=opt.dataset_root_dir, save_path=opt.save_path, 
             resume_path2d=opt.resume_path2d, resume_path3d=opt.resume_path3d, 
             attention=attention, patchnv=patchnv, debug=debug)
    print("Evaluation Done!")