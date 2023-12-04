#!/usr/bin/env python
from __future__ import print_function

import argparse
import configparser
import math
import os
import random
import shutil
from datetime import datetime
from os import makedirs
from os.path import isfile, join

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import model3d.PointNetVlad as PNV
from crossmodal.models.models_generic import get_backend, get_model
from crossmodal.tools.datasets import input_transform
from crossmodal.training_tools.get_clusters import get_clusters
from crossmodal.training_tools.tools import save_checkpoint
from mycode.msls import MSLS
from mycode.NetVLAD.netvlad import get_model_netvlad
from mycode.train_epoch import train_epoch, train_epoch_no_mining
from mycode.val import val
from sphereModel.sphereresnet import (sphere_resnet18, sphere_resnet34,
                                      sphere_resnet50)

# single GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# multi GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def get_learning_rate(epoch):
    
    learning_rate = 0.0001 * ((0.8) ** (epoch // 2))  # 0.00005
    # learning_rate = max(learning_rate, 0.00001) * 50  # CLIP THE LEARNING RATE!
    return learning_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CrossModal-train')
    parser.add_argument('--config_path', type=str, default='crossmodal/configs/train.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data')
    parser.add_argument('--cache_path', type=str, default='/data-lyh/KITTI360/cache',
                        help='Path to save cache, centroid data to.')
    parser.add_argument('--save_path', type=str, default='/log/checkpoints', required=True,
                        help='Path to save checkpoints to')
    parser.add_argument('--resume_path2d', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for 2d resuming training.') # /home/zhipengz/result2/Aug26_15-48-13_vgg_clu64_4/checkpoints/model_best.pth.tar
    parser.add_argument('--resume_path3d', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for 3d resuming training.') # /home/zhipengz/result2/Aug26_15-48-13_vgg_clu64_4/checkpoints3d/model_best.ckpt
    parser.add_argument('--cluster_path', type=str, default='',
                        help='Full path and name (with extension) to load cluster data from, for resuming training.')# /data/zzp/cache/centroids/vgg_20m_KITTI360_64_desc_cen.hdf5
    parser.add_argument('--dataset_root_dir', type=str, default='/data-lyh/KITTI360', required=True,
                        help='Root directory of dataset')
    parser.add_argument('--id', type=str, default='vgg', required=True,
                        help='Description of this model, e.g. vgg16_netvlad')
    parser.add_argument('--nEpochs', type=int, default=50, 
                        help='number of epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_every_epoch', action='store_true', 
                        help='Flag to set a separate checkpoint file for each new epoch')
    parser.add_argument('--threads', type=int, default=16, 
                        help='Number of threads for each data loader to use')
    parser.add_argument('--nocuda', action='store_true', 
                        help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--attention', action='store_true', 
                        help='If true, add SE attention to backbone.')
    parser.add_argument('--network', type=str, default='resnet', 
                        help='2D CNN network, e.g. vgg,resnet,spherical')
    parser.add_argument('--pretrained_cnn_network', type=bool, default=False,
                        help='whether use pretrained 2D CNN network')
    parser.add_argument('--debug', action='store_true', 
                        help='whether debug mode.')
    
    opt = parser.parse_args()
    print(opt)
    # load basic train parameters
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)
    print('os.environ[CUDA_VISIBLE_DEVICES]:\t',os.environ['CUDA_VISIBLE_DEVICES'])
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    print(device)
    # random seed
    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        torch.cuda.manual_seed(int(config['train']['seed']))
    print('===> Building 2d model')
    # attention switcher
    attention = False
    if opt.attention:
        attention = True
    print('attention usage:\t', attention)
    print('Current 2D CNN network backbone:\t', opt.network)
    # feature extract network
    pre = opt.pretrained_cnn_network
    print('whether use pretrained 2D CNN network:\t', opt.pretrained_cnn_network)
    # basic backbone
    if opt.network == 'spherical':
        encoder = sphere_resnet34(pretrained=pre)
        encoder_dim = 512
    elif opt.network == 'resnet':
        encoder_dim, encoder = get_backend(net='resnet', pre=pre)
    elif opt.network == 'vgg':
        encoder_dim, encoder = get_backend(net='vgg', pre=pre)
    else:
        raise ValueError('Unknown cnn network')
    # if already started training earlier and continuing
    if opt.resume_path2d:
        if isfile(opt.resume_path2d):
            print("===> loading checkpoint '{}'".format(opt.resume_path2d))
            checkpoint = torch.load(opt.resume_path2d, map_location=lambda storage, loc: storage)
            try:
                config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])
            except:
                # for the resume training the new model is adjust with a new module class by zhipeng
                config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.module.centroids'].shape[0])         
            # load same image bachbone
            model = get_model_netvlad(encoder, encoder_dim, config['global_params'], attention=attention)
            # load parameters
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("===> loaded checkpoint '{}'".format(opt.resume_path2d))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(opt.resume_path2d))
    else:  # if not, assume fresh training instance and will initially generate cluster centroids
        print('===> Building clusters')
        config['global_params']['num_clusters'] = config['train']['num_clusters']
        model = get_model_netvlad(encoder, encoder_dim, config['global_params'], attention=attention)
        # so what this cluster used for? code similar to Patch-NetVLAD code
        initcache = join(opt.cache_path, 'centroids', opt.network + '_20m_KITTI360_' + config['train']['num_clusters'] + '_desc_cen.hdf5')
        print('initcache:\t', initcache)
        # predefined cluster centers, have little effect to final results
        if opt.cluster_path:
            if isfile(opt.cluster_path):
                if opt.cluster_path != initcache:
                    shutil.copyfile(opt.cluster_path, initcache)
            else:
                raise FileNotFoundError("=> no cluster data found at '{}'".format(opt.cluster_path))
        else:
            print('===> Finding cluster centroids')
            print('===> Loading dataset(s) for clustering')
            train_dataset = MSLS(opt.dataset_root_dir, mode='train', transform=input_transform(train=False),
                                 batch_size=int(config['train']['cachebatchsize']), threads=opt.threads, margin=float(config['train']['margin']))
            print(train_dataset)
            model = model.to(device)
            print('===> Calculating descriptors and clusters')
            get_clusters(train_dataset, model, encoder_dim, device, opt, config, initcache)
            # a little hacky, but needed to easily run init_params
            model = model.to(device="cpu")

        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pool.init_params(clsts, traindescs)
            del clsts, traindescs

    isParallel = False
    if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        model.attention = nn.DataParallel(model.attention)
        model.pool = nn.DataParallel(model.pool)
        isParallel = True
    # lr
    optimizer2d = None
    scheduler = None
    if config['train']['optim'] == 'ADAM':
        optimizer2d = optim.Adam(filter(lambda par: par.requires_grad, model.parameters()), lr=float(config['train']['lr']))  # , betas=(0,0.9))
    elif config['train']['optim'] == 'SGD':
        optimizer2d = optim.SGD(filter(lambda par: par.requires_grad, model.parameters()), lr=float(config['train']['lr']),
                              momentum=float(config['train']['momentum']), weight_decay=float(config['train']['weightDecay']))
        scheduler = optim.lr_scheduler.StepLR(optimizer2d, step_size=int(config['train']['lrstep']), gamma=float(config['train']['lrgamma']))
    else:
        raise ValueError('Unknown optimizer2d: ' + config['train']['optim'])

    model = model.to(device)
    if opt.resume_path2d:
        optimizer2d.load_state_dict(checkpoint['optimizer'])
    # 3D encoder
    print('===> Building 3d model')
    if attention:
        # attention mechanism by AE-Spherical
        model3d = PNV.PointNetVlad_attention(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
        model3d.attention.init_weights()
    else:
        # vanilla PointNetVLAD
        model3d = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    # print('model3d:\t', model3d)
    model3d = model3d.to(device)
    if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model3d = nn.DataParallel(model3d)
    # 3D learning rate
    learning_rate = get_learning_rate(opt.start_epoch)
    print('3dLR:\t', learning_rate)
    # retain the specific layers to be optimized(require gradients and fix others)
    parameters3d = filter(lambda p: p.requires_grad, model3d.parameters())
    optimizer3d = optim.Adam(parameters3d, learning_rate)
    # scheduler3d = torch.optim.lr_scheduler.LambdaLR(optimizer3d, get_learning_rate, last_epoch=-1)
    if opt.resume_path3d:
        print("=> loading 3d model '{}'".format(opt.resume_path3d))
        checkpoint3d = torch.load(opt.resume_path3d)
        print("chepoint3d:\t", checkpoint3d.keys())
        model3d.load_state_dict(checkpoint3d['state_dict'], strict=False)
    # triplet loss
    # more hints at:https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # A nonnegative margin representing the minimum difference between the positive and negative distances required for the loss to be 0
    criterion = nn.TripletMarginLoss(margin=float(config['train']['margin']) ** 0.5, p=2, reduction='sum').to(device)
    # datasets
    print('===> Loading dataset(s)')
    train_dataset = MSLS(opt.dataset_root_dir, mode='train', nNeg=int(config['train']['nNeg']),
                         transform=input_transform(train=True),
                         batch_size=int(config['train']['batchsize']), threads=opt.threads,
                         margin=float(config['train']['margin']))
    if not train_dataset.mining:
        train_dataset.generate_triplets()
        training_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                        batch_size=int(config['train']['batchsize']), shuffle=True, persistent_workers=True,
                                        collate_fn=MSLS.collate_fn, pin_memory=True)
    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', transform=input_transform(train=False),
                              batch_size=int(config['train']['batchsize']), threads=opt.threads,
                              margin=float(config['train']['margin']))
    print('===> Training set, query number:', len(train_dataset.qIdx))
    print('===> Validation set, query number:', len(validation_dataset.qIdx))
    print('===> Begin to training the AE-Spherical model.')
    # summary
    writer = SummaryWriter(log_dir=join(opt.save_path, datetime.now().strftime('%b%d_%H_%M_%S') + '_' + opt.id))
    # write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    print('logdir:\t',logdir)
    opt.save_file_path2d = join(logdir, 'checkpoints')
    makedirs(opt.save_file_path2d)
    opt.save_file_path3d = join(logdir, 'checkpoints3d')
    makedirs(opt.save_file_path3d)
    # precision initialization
    not_improved = 0
    best_score = 0
    if opt.resume_path2d:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']
    # save initial weight bachbones
    if opt.debug:
        recalls = {1:0.1, 5:0.2, 10:0.5, 20:0.6, 50:0.9}
        save_checkpoint({
                    'epoch': 0,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'not_improved': not_improved,
                    'optimizer': optimizer2d.state_dict(),
                    'parallel': isParallel,
                }, opt, False, filename='checkpoint_initial.pth.tar')    
        # save 3d
        torch.save({
            'epoch': 0,
            'state_dict': model3d.state_dict(),
            'optimizer': optimizer3d.state_dict(),
        }, opt.save_file_path3d + "/" + "model_initial.ckpt")
    # loop train
    for epoch in trange(opt.start_epoch + 1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        if train_dataset.mining:
            train_epoch(train_dataset, model, model3d, optimizer2d, optimizer3d, criterion, 
                        encoder_dim, device, epoch, opt, config, writer)
        else:
            train_epoch_no_mining(train_dataset, training_data_loader, model, model3d, optimizer2d, 
                                  optimizer3d, criterion, epoch, config, writer)
        # 2d model learning rate decay is based on epoch number
        if scheduler is not None:
            #scheduler.step(epoch) may be the epoch will be deprecated in the future?
            scheduler.step()
        # learning rate decay for 3d model
        lr_3d = get_learning_rate(epoch)
        parameters3d = filter(lambda p: p.requires_grad, model3d.parameters())
        optimizer3d = optim.Adam(parameters3d, lr_3d)
        # validation
        if (epoch % int(config['train']['eval_every'])) == 0:
            print("Validation begins at epoch: ", epoch)
            recalls = val(validation_dataset, model, model3d, device, opt.threads, config, writer, epoch, write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1
            # save checkpoint
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer2d.state_dict(),
                'parallel': isParallel,
            }, opt, is_best)
            # fake multi-GPU training
            if isinstance(model3d, nn.DataParallel):
                model_to_save = model3d.module
            else:
                model_to_save = model3d
            # save 3d
            save_name = opt.save_file_path3d + "/" + "model.ckpt"
            torch.save({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer3d.state_dict(),
            }, save_name)
            if is_best:
                shutil.copyfile(save_name, join(opt.save_file_path3d, 'model_best.ckpt'))
            # early stoping will be done when the non-improvement epoch has already exceed the patience tolerance value
            if int(config['train']['patience']) > 0 and not_improved > (
                    int(config['train']['patience']) / int(config['train']['eval_every'])):
                print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
                break
            print("Validation finished!")

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()
    # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    torch.cuda.empty_cache()  
    # memory after runs
    print('Done')
