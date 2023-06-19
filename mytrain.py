#!/usr/bin/env python
from __future__ import print_function

import argparse
import configparser
import os
import random
import shutil
import math
from os.path import join, isfile
from os import makedirs
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import h5py
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import trange

from mycode.NetVLAD.netvlad import get_model_netvlad
import model3d.PointNetVlad as PNV
from sphereModel.sphereresnet import sphere_resnet18
from mycode.msls import MSLS

from mycode.train_epoch import train_epoch
from mycode.val import val
from crossmodal.training_tools.get_clusters import get_clusters
from crossmodal.training_tools.tools import save_checkpoint
from crossmodal.tools.datasets import input_transform
from crossmodal.models.models_generic import get_backend, get_model

# single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# multi GPUs
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(self.min_lr, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

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
    parser.add_argument('--threads', type=int, default=0, 
                        help='Number of threads for each data loader to use')
    parser.add_argument('--nocuda', action='store_true', 
                        help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--attention', action='store_true', 
                        help='If true, add SE attention to backbone.')
    parser.add_argument('--network', type=str, default='resnet', 
                        help='2D CNN network, e.g. vgg,resnet,spherical')
    parser.add_argument('--pretrained_cnn_network', action='store_true', required=True,
                        help='whether use pretrained 2D CNN network')
    # multi GPUs setting
    #parser.add_argument("--local_rank", type=int)
    # parser.add_argument('--workers', default=32, type=int, metavar='N',
    #                     help='number of data loading workers (default: 32)')
    # parser.add_argument('--world_size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                      'N processes per node, which has N GPUs. This is the '
    #                      'fastest way to use PyTorch for either single node or '
    #                      'multi node data parallel training')
    
    opt = parser.parse_args()
    print(opt)
    # load basic train parameters
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    # multi GPUs setting
    print('os.environ[CUDA_VISIBLE_DEVICES]:\t',os.environ['CUDA_VISIBLE_DEVICES'])
    #torch.cuda.set_device(opt.local_rank) 
    #torch.distributed.init_process_group(backend='nccl')
    # device_ids = [0, 1, 2, 3]
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
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
        encoder = sphere_resnet18(pretrained=pre)
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
                                 batch_size=int(config['train']['batchsize']), threads=opt.threads, margin=float(config['train']['margin']))
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
        model.pool = nn.DataParallel(model.pool)
        # model3d = nn.DataParallel(model3d)
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
    elif config['train']['optim'] == 'Warmup':
        optimizer2d = optim.SGD(filter(lambda par: par.requires_grad, model.parameters()), lr=float(config['train']['lr']),
                              momentum=float(config['train']['momentum']), weight_decay=float(config['train']['weightDecay']))
        scheduler = WarmupLinearSchedule(optimizer2d, warmup_steps=0.03, t_total=opt.nEpochs)
    else:
        raise ValueError('Unknown optimizer2d: ' + config['train']['optim'])

    model = model.to(device)
    print('model2d:\t', model)
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
    print('model3d:\t',model3d)
    model3d = model3d.to(device)
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
                         batch_size=int(config['train']['cachebatchsize']), threads=opt.threads,
                         margin=float(config['train']['margin']))
    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', transform=input_transform(train=False),
                              batch_size=int(config['train']['cachebatchsize']), threads=opt.threads,
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
    # loop train
    for epoch in trange(opt.start_epoch + 1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        train_epoch(train_dataset, model, model3d, optimizer2d, optimizer3d, criterion, 
                    encoder_dim, device, epoch, opt, config, writer)
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
            recalls = val(validation_dataset, model, model3d, 
                          encoder_dim, device, opt.threads, config, writer, epoch,
                          write_tboard=True, pbar_position=1)
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
