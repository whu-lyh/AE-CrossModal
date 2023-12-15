import os
import numpy as np
from tqdm import trange, tqdm
import torch
import torch.distributed
import torch.nn as nn
from torch.utils.data import DataLoader
from crossmodal.training_tools.tools import humanbytes
from mycode.msls import MSLS

pdist = nn.PairwiseDistance(p=2)
debug = False

def train_iteration(train_dataset, training_data_loader, startIter, model2d, model3d, criterion, optimizer, optimizer3d, epoch_loss, epoch_num, nBatches, writer):
    # accumulate the loss, making sure the loss is stable at smaller batch_size(e.g. 2)
    # this operation makes the batch size equals to accum_steps times of the raw value
    accum_steps = 16
    # calculate loss per query triplet in batch
    for iteration, (query, query_pc, positives, positives_pc, negatives, negatives_pcs, negCounts, indices) in \
            enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
        # where N = batchSize * (nQuery + nPos + nNeg)
        B = query.shape[0]
        if debug:
            batch1 = {}
            # save query, positive, negatives file name
            batch1['query'] = os.path.basename(train_dataset.qImages[indices[0]])
            batch1['positive'] = os.path.basename(train_dataset.dbImages[indices[1]])
            batch1['negatives'] = [os.path.basename(train_dataset.dbImages[indices[i]]) for i in range(2,7)]
            print('batch1:\t',batch1)
        # negative images in this batch
        nNeg = torch.sum(negCounts)
        # glob 2d data(tensor) to 2d model
        input_tuplet_img = torch.cat([query, positives, negatives])
        input_tuplet_img = input_tuplet_img.cuda()
        image_encoding = model2d.encoder(input_tuplet_img)
        output_feat_img = model2d.pool(image_encoding)
        # there are 5 negatives for 2d image
        global_feat_img_query, global_feat_img_pos, global_feat_img_negs = torch.split(output_feat_img, [B, B, B*5])
        # glob 3d data(tensor) into 3d model
        # query_pc = query_pc.float()
        # positives_pc = positives_pc.float()
        # negatives_pcs = negatives_pcs.float()
        input_tuplet_pc = torch.cat((query_pc, positives_pc, negatives_pcs), 1)
        input_tuplet_pc = input_tuplet_pc.view((-1, 1, 4096, 3))
        # input_tuplet_pc.requires_grad_(True)
        input_tuplet_pc = input_tuplet_pc.cuda()
        output_feat_pc = model3d(input_tuplet_pc)
        # 256 is feature dimension
        output_feat_pc = output_feat_pc.view(B , -1, 256)
        global_feat_pc_query, global_feat_pc_pos, global_feat_pc_negs = torch.split(output_feat_pc, [1, 1, 5], dim=1)
        global_feat_pc_query = global_feat_pc_query.view(-1, 256)
        global_feat_pc_pos = global_feat_pc_pos.view(-1, 256)
        global_feat_pc_negs = global_feat_pc_negs.contiguous().view(-1, 256)
        # calculate loss for each Query, Positive, Negative triplet
        # due to potential difference in number of negatives have to
        # do it per query, per negative
        loss_dic = {}
        loss_recode = {}
        loss_je = 0
        loss_2dto3d = 0
        loss_3dto2d = 0
        loss_2dto2d = 0
        loss_3dto3d = 0
        # negCounts is the number of negative samples, all negatives will participate the loss calculation
        # infact there is not necessary to calculate the loss seperately
        # the TripletMarginLoss support batch operation
        for i, negCount in enumerate(negCounts):
            # feature distance between query and database
            loss_je += pdist(global_feat_img_query[i: i + 1] , global_feat_pc_query[i: i + 1])
            for n in range(negCount):
                negIx = (torch.sum(negCounts[:i]) + n).item()
                # triplet loss under 4 modes, here infact only 1 negative sample is used
                loss_2dto3d += criterion(global_feat_img_query[i: i + 1], global_feat_pc_pos[i: i + 1], global_feat_pc_negs[negIx:negIx + 1])
                loss_3dto2d += criterion(global_feat_pc_query[i: i + 1], global_feat_img_pos[i: i + 1], global_feat_img_negs[negIx:negIx + 1])
                loss_2dto2d += criterion(global_feat_img_query[i: i + 1], global_feat_img_pos[i: i + 1], global_feat_img_negs[negIx:negIx + 1])
                loss_3dto3d += criterion(global_feat_pc_query[i: i + 1], global_feat_pc_pos[i: i + 1], global_feat_pc_negs[negIx:negIx + 1])
                if debug:
                    loss_recode['2dto3d'] = loss_2dto3d.data
                    loss_recode['3dto2d'] = loss_3dto2d.data
                    loss_recode['2dto2d'] = loss_2dto2d.data
                    loss_recode['3dto3d'] = loss_3dto3d.data
                    print('loss_recode:', loss_recode)
        # sum loss
        loss_sm = loss_2dto2d + loss_3dto3d
        loss_cm = loss_2dto3d + loss_3dto2d
        # weighted loss
        loss = 0.1 * loss_sm + loss_cm + loss_je
        # the total loss in this batch, the negative sample number should be used!!!
        loss /= nNeg.float().cuda()
        loss_je_t = loss_je / nNeg.float().cuda()
        loss_cm_t = loss_cm / nNeg.float().cuda()
        loss_sm_t = loss_sm / nNeg.float().cuda()
        loss = loss / accum_steps
        # calculate gradient
        loss.backward()
        # clean the gradient when accumulated some batches, avoiding the smaller batch_size
        if (iteration + 1) % accum_steps == 0 or (iteration + 1) == len(training_data_loader):
            optimizer.step()
            optimizer.zero_grad()
            optimizer3d.step()
            # zero out gradients so we can accumulate new ones over batches
            optimizer3d.zero_grad()
        # release memory
        del input_tuplet_img, image_encoding, output_feat_img, global_feat_img_query, global_feat_img_pos, global_feat_img_negs
        del input_tuplet_pc, output_feat_pc, global_feat_pc_query, global_feat_pc_pos, global_feat_pc_negs
        del query, query_pc, positives, positives_pc, negatives, negatives_pcs
        # loss in current batch
        batch_loss = loss.item() * accum_steps
        epoch_loss += batch_loss

        if iteration % 100 == 0 or nBatches <= 10:
            tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration, nBatches, batch_loss))
            # weighted loss in this batch
            writer.add_scalar('Train/Loss', batch_loss, ((epoch_num - 1) * nBatches) + iteration)
            # directly querys from two modalities
            writer.add_scalar('Train/Loss_cm_query', loss_je_t.item(), ((epoch_num - 1) * nBatches) + iteration)
            # cross modality loss
            writer.add_scalar('Train/Loss_cm_triplet', loss_cm_t.item(), ((epoch_num - 1) * nBatches) + iteration)
            # same modality loss
            writer.add_scalar('Train/Loss_sm_triplet', loss_sm_t.item(), ((epoch_num - 1) * nBatches) + iteration)
            # the total negatives in this batch
            writer.add_scalar('Train/nNeg', nNeg, ((epoch_num - 1) * nBatches) + iteration)
            #tqdm.write("GPU Allocated:\t", humanbytes(torch.cuda.memory_allocated()))
            #tqdm.write("GPU Cached:\t", humanbytes(torch.cuda.memory_reserved()))
        del loss


def train_epoch(train_dataset, model2d, model3d, optimizer, optimizer3d, criterion, encoder_dim, device, epoch_num, opt, config, writer):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
    # forward neural networks, open BN and Droupout module
    model2d.train()
    model3d.train()
    # the number of total batches during in this epoch, each batch will contain batch_size samples
    nBatches = (len(train_dataset.qIdx) + int(config['train']['batchsize']) - 1) // int(config['train']['batchsize'])
    # initialize loss
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging
    tqdm.write('Number of total batches:\t' + str(nBatches))
    tqdm.write('Number of triplets:\t' + str(nBatches * int(config['train']['batchsize'])))
    if not train_dataset.mining:
        training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                            batch_size=int(config['train']['batchsize']), shuffle=True, persistent_workers=True,
                                            collate_fn=MSLS.collate_fn, pin_memory=cuda)
        train_iteration(train_dataset, training_data_loader, startIter, model2d, model3d, criterion, optimizer, optimizer3d, epoch_loss, epoch_num, nBatches, writer)
        del training_data_loader
        optimizer.zero_grad()
        optimizer3d.zero_grad()
        torch.cuda.empty_cache()
    else:       
        # shuffle new samples to be trained
        train_dataset.new_epoch()
        # train_dataset.nCacheSubset is number of the subsets in one single epoch
        # each batch will be optimized during each cached subset, while the subset data are randomly selected
        for subIter in trange(train_dataset.nCacheSubset, desc='Training...SubCache refreshing'.rjust(15), position=1):
            # global feature dimension, by default 256
            pool_global_feature_dim = 256
            print("====> Building Cache")
            # generate triplets for training, here absolutely positive and negative tuplets are sampled
            # if epoch_num > 10:
            #     train_dataset.update_subcache(net=model2d, net3d=model3d, outputdim=pool_global_feature_dim)
            # else:
            train_dataset.update_subcache(net=None, net3d=None, outputdim=pool_global_feature_dim)
            # this online manner cost much more time, do not use
            # train_dataset.update_subcache(net=model2d, net3d=model3d, outputdim=pool_global_feature_dim)
            # add train triplet dataset into dataloader, batch triplets will be loaded
            training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                            batch_size=int(config['train']['batchsize']), shuffle=True, persistent_workers=True,
                                            collate_fn=MSLS.collate_fn, pin_memory=cuda)

            train_iteration(train_dataset, training_data_loader,startIter, model2d, model3d,criterion,optimizer, optimizer3d, epoch_loss, epoch_num, nBatches, writer)
            # start iteration in whole epoch, increase at batch_size step
            startIter += len(training_data_loader)
            del training_data_loader
            optimizer.zero_grad()
            optimizer3d.zero_grad()
            torch.cuda.empty_cache()

    # average loss in current epoch with whole batch iterations
    avg_loss = epoch_loss / nBatches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)


def train_epoch_no_mining(train_dataset, training_data_loader, model2d, model3d, optimizer, optimizer3d, criterion, epoch_num, config, writer):
    # forward neural networks, open BN and Droupout module
    model2d.train()
    model3d.train()
    # the number of total batches during in this epoch, each batch will contain batch_size samples
    nBatches = (len(train_dataset.qIdx) + int(config['train']['batchsize']) - 1) // int(config['train']['batchsize'])
    # initialize loss
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging
    tqdm.write('Number of total batches:\t' + str(nBatches))
    tqdm.write('Number of triplets:\t' + str(nBatches * int(config['train']['batchsize'])))
    train_iteration(train_dataset, training_data_loader, startIter, model2d, model3d, criterion, optimizer, optimizer3d, epoch_loss, epoch_num, nBatches, writer)
    optimizer.zero_grad()
    optimizer3d.zero_grad()
    torch.cuda.empty_cache()   
    # average loss in current epoch with whole batch iterations
    avg_loss = epoch_loss / nBatches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
