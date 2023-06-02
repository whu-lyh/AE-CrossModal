import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random
import sys
import itertools
from tqdm import tqdm
import re
from mycode.loading_pointclouds import load_pc_file, load_pc_files
import random

# 03 is too short to find triplets
default_cities = {
    'train': ["0", "2", "4", "5", "6", "7", "9", "10"],
    'val': ["3"],
    'test': []
}


class ImagesFromList(Dataset):
    '''
        return a np array of the idx-th image
    '''
    def __init__(self, images, transform):
        self.images_list = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # get idx image into np.array
        img_dir = self.images_list[idx]
        img = Image.open(img_dir)
        img = self.transform(img)
        return img, idx


class PcFromFiles(Dataset):
    '''
        return a np array of the idx-th point cloud
    '''
    def __init__(self, pcs):
        self.pcs_list = np.asarray(pcs)

    def __len__(self):
        return len(self.pcs_list)

    def __getitem__(self, idx):
        pc = load_pc_file(self.pcs_list[idx])
        # no other argumentation operation
        return pc, idx


class MSLS(Dataset):
    def __init__(self, root_dir, cities='', nNeg=5, transform=None, mode='train', 
                 posDistThr=15, negDistThr=35, cached_queries=4000, cached_negatives=25000,
                 batch_size=2, threads=8, margin=0.2):
        # initializing
        assert mode in ('train', 'val', 'test')
        # check sequences to be processed
        if cities in default_cities:
            self.cities = default_cities[cities]
        elif cities == '':
            self.cities = default_cities[mode]
        else:
            self.cities = cities.split(',')
        # image and submap share the same idx, initialized as list, but turn into np.array after initialization
        self.qIdx = [] # query index
        self.qImages = [] # query images' path
        self.qPcs = [] # query pcs' path
        self.dbImages = [] # database images path
        self.dbPcs = [] # database pcs path
        # index
        self.pIdx = [] # postive index
        self.nonNegIdx = [] # neg
        self.qEndPosList = []
        self.dbEndPosList = []
        self.all_pos_indices = [] # gt
        # hyper-parameters
        self.nNeg = nNeg # negative number
        self.margin = margin # triplet margin
        self.posDistThr = posDistThr # posetive distance
        self.negDistThr = negDistThr # negative distance
        self.cached_queries = cached_queries # cached queries
        self.cached_negatives = cached_negatives # cached negatives
        # flags
        self.cache = None
        self.mode = mode
        # other
        self.transform = transform
        # load data
        for city_idx in self.cities:
            city='2013_05_28_drive_%04d_sync' % int(city_idx)
            print("=====> {}".format(city))
            subdir_img = 'data_2d_pano'
            subdir_submap = 'data_3d_submap'
            # get len of images from cities so far for global indexing
            _lenQ = len(self.qImages)
            _lenDb = len(self.dbImages)
            # when GPS / UTM is available, if there is no overlaps between train and val, the following loading code has no difference
            if self.mode in ['train', 'val']:
                # load query and database data
                if self.mode == 'val':
                    qData = pd.read_csv(join(root_dir, subdir_img, city, 'query.csv'), index_col=0)
                    # load database data
                    dbData = pd.read_csv(join(root_dir, subdir_img, city, 'database.csv'), index_col=0)
                else:
                    qData = pd.read_csv(join(root_dir, subdir_img, city, 'query.csv'), index_col=0)
                    # load database data
                    dbData = pd.read_csv(join(root_dir, subdir_img, city, 'database.csv'), index_col=0)
                # what is the usage of the seq structure? or just some inherit from MSLS, bingo
                # fetch query data, specifically data path
                qSeqIdxs, qSeqKeys, qSeqKeys_pc = self.arange_as_seq(qData, 
                                                                    join(root_dir, subdir_img, city), 
                                                                    join(root_dir, subdir_submap, city))
                # load database data
                dbSeqIdxs, dbSeqKeys, dbSeqKeys_pc = self.arange_as_seq(dbData, 
                                                                    join(root_dir, subdir_img, city), 
                                                                    join(root_dir, subdir_submap, city))
                # if there are no query/dabase images,
                # then continue to next city
                if len(qSeqIdxs) == 0 or len(dbSeqIdxs) == 0:
                    continue
                # here qImages is same as qSeqKeys, this kind of operation is designed for MSLS sequence retrieval task especially
                self.qImages.extend(qSeqKeys)
                self.qPcs.extend(qSeqKeys_pc)
                self.dbImages.extend(dbSeqKeys)
                self.dbPcs.extend(dbSeqKeys_pc)
                self.qEndPosList.append(len(qSeqKeys))
                self.dbEndPosList.append(len(dbSeqKeys))
                print('self.qEndPosList:\t',self.qEndPosList)
                print('self.dbEndPosList:\t',self.dbEndPosList)
                # utm coordinates, vital for training and validation
                utmQ = qData[['east', 'north']].values.reshape(-1, 2)
                utmDb = dbData[['east', 'north']].values.reshape(-1, 2)
                # find positive images for training and testing
                # for all query images
                neigh = NearestNeighbors(algorithm='brute')
                neigh.fit(utmDb)
                pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
                # the nearest idxes will be the ground truth when val mode
                self.all_pos_indices.extend(pos_indices)
                # fetch negative pairs for triplet turple, but the negatives here contains the positives
                if self.mode == 'train':
                    nD, negIdx = neigh.radius_neighbors(utmQ, self.negDistThr)
                # get all idx unique in whole dataset
                for q_seq_idx in range(len(qSeqIdxs)):
                    q_frame_idxs = q_seq_idx
                    q_uniq_frame_idx = q_frame_idxs
                    p_uniq_frame_idxs = pos_indices[q_uniq_frame_idx]
                    # the query image has at least one positive
                    if len(p_uniq_frame_idxs) > 0:
                        p_seq_idx = np.unique(dbSeqIdxs[p_uniq_frame_idxs])
                        # qIdx contains whole sequences, and the index is unique to whole (training or validation) datasets
                        self.qIdx.append(q_seq_idx + _lenQ)
                        self.pIdx.append(p_seq_idx + _lenDb)
                        # in training we have two thresholds, one for finding positives and one for finding images
                        # that we are certain are negatives.
                        if self.mode == 'train':
                            # n_uniq_frame_idxs = [n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg]
                            n_uniq_frame_idxs = negIdx[q_uniq_frame_idx]
                            n_seq_idx = np.unique(dbSeqIdxs[n_uniq_frame_idxs])
                            self.nonNegIdx.append(n_seq_idx + _lenDb)

            elif self.mode in ['test']:
                qData = pd.read_csv(join(root_dir, subdir_img, city, 'query.csv'), index_col=0)
                # load database data
                dbData = pd.read_csv(join(root_dir, subdir_img, city, 'database.csv'), index_col=0)
                # fetch query data, specifically data path
                qSeqIdxs, qSeqKeys, qSeqKeys_pc = self.arange_as_seq(qData, 
                                                                    join(root_dir, subdir_img, city),
                                                                    join(root_dir, subdir_submap, city))
                # load database data
                dbSeqIdxs, dbSeqKeys, dbSeqKeys_pc= self.arange_as_seq(dbData, 
                                                                    join(root_dir, subdir_img, city),
                                                                    join(root_dir, subdir_submap, city))
                # here qImages is same as qSeqKeys, this kind of operation is designed for MSLS sequence retrieval task especially
                self.qImages.extend(qSeqKeys)
                self.dbImages.extend(dbSeqKeys)
                self.qPcs.extend(qSeqKeys_pc)
                self.dbPcs.extend(dbSeqKeys_pc)
        # whole sequence datas are gathered for batch optimization
        # Note that the number of submap is same as the number of images
        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print("Exiting...")
            print("there are no query/database images.")
            print("Try more sequences")
            sys.exit()
        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.pIdx = np.asarray(self.pIdx)
        self.nonNegIdx = np.asarray(self.nonNegIdx)
        # self.qIdx = np.asarray(self.qIdx,dtype=object) # might have some wired bugs, warning is acceptable
        # self.pIdx = np.asarray(self.pIdx,dtype=object)
        # self.nonNegIdx = np.asarray(self.nonNegIdx,dtype=object)
        # here only data path is stored
        self.qImages = np.asarray(self.qImages)
        self.qPcs = np.asarray(self.qPcs)
        self.dbImages = np.asarray(self.dbImages)
        self.dbPcs = np.asarray(self.dbPcs)

        self.device = torch.device("cuda")
        self.threads = threads
        self.batch_size = batch_size


    @staticmethod
    def arange_as_seq(data, path_img, path_pc):
        '''
            arrange all query data(images, submaps) into list container
            Return：
                idx in csv file, image path, and pc full path in list container
        '''
        seq_keys, seq_idxs, seq_keys_pc = [], [], []
        for seq_idx in data.index:
            # find surrounding frames in sequence
            # iloc is a function of pandas library for get the seq_idx record
            seq = data.iloc[seq_idx]
            img_num = int(re.sub('[a-z]', '', seq['key']))
            seq_key = join(path_img, 'pano', 'data_rgb', '%010d' % img_num + '.png')
            seq_key_pc = join(path_pc, 'submaps', '%010d' % img_num + '.bin')
            # append into list
            seq_keys.append(seq_key)
            seq_keys_pc.append(seq_key_pc)
            seq_idxs.append([seq_idx])
        return np.asarray(seq_idxs), seq_keys, seq_keys_pc

    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):
        '''
            Purpose: 
                slice the whole query data into cached subsets based on the cached_queries number
                reset the subcache data from all query indices, shuffle is utilized
                random query subcache samples will be generated, whole query data will be forworded in a subset unit
            Note:
                Whole sequences data will be globed together into self.qIdx
        '''
        # find how many subsets we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)
        # get all query indices
        arr = list(range(len(self.qIdx)))
        random.shuffle(arr)
        arr = np.array(arr)
        # the subcached_indices will be extracted from shuffled qIdx
        # the whole query data will be divided into subsets using self.cached_queries as interval
        # subcache_indices contains the query data idx in current subset, and covers whole sequences in training datasets
        # to be more aggressive, if cached_queries is smaller than 
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)
        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net=None, net3d=None, outputdim=None):
        '''
            Purpose: get self.triplets fufilled from current subset
            Specifically,   get the query idxs from cached subset, the query data is randomly selected each epoch
                            get its postive and negatives
        '''
        # reset triplets
        self.triplets = []
        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        if self.current_subset >= len(self.subcache_indices):
            tqdm.write('Reset epoch - FIX THIS LATER!')
            self.current_subset = 0
        # take n (query,positive,negatives) triplet images from current cached subsets
        qidxs = np.asarray(self.subcache_indices[self.current_subset])
        #print("len(qidxs):\t", qidxs) # should be same as self.cached_queries (or slightly smaller than, for the last subset)
        # build triplets based on randomly selection from data
        if net is None and net3d is None:
            for q in qidxs:
                # get query idx
                qidx = self.qIdx[q]
                # get positives
                pidxs = self.pIdx[q]
                # choose a random positive (within positive range default self.posDistThr m)
                pidx = np.random.choice(pidxs, size=1)[0]
                # get negatives, 5 by default
                while True:
                    # randomly select negatives from whole sequences in training dataset
                    nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)
                    # FIXME: check whether the negaitves fall inside 20 threshold
                    # ensure that non of the choice negative images are within the negative range (default 25 m)
                    if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                        break
                # package the triplet and target, all the indices are the indicex of csv file
                triplet = [qidx, pidx, *nidxs]
                target = [-1, 1] + [0] * len(nidxs)
                self.triplets.append((triplet, target))
            # increment subset counter
            self.current_subset += 1
            return
        # take their 5 positives in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in np.random.choice(idx, size=5, replace=False)])
        # print('pidxs:\t', pidxs)
        #print('len(pidxs):\t', len(pidxs))
        nidxs = []
        while len(nidxs) < self.cached_queries // 10:
            # take m = 5*cached_queries is number of negative images
            nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)
            # and make sure that there is no positives among them
            nidxs = nidxs[np.in1d(nidxs, np.unique(
                [i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]
            #print('len(nidxs2):\t', len(nidxs))
        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.batch_size, 'shuffle': False, 'persistent_workers': True, 
               'num_workers': self.threads, 'pin_memory': True}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform), **opt)
        ploader_pc = torch.utils.data.DataLoader(PcFromFiles(self.dbPcs[pidxs]), **opt)
        nloader_pc = torch.utils.data.DataLoader(PcFromFiles(self.dbPcs[nidxs]), **opt)
        # calculate their descriptors
        net.eval()
        net3d.eval()
        with torch.no_grad():
            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device) # all query
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device) # all corresponding positives
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device) # all corresponding negatives
            batch_size = opt['batch_size']
            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), 
                                 desc='compute query descriptors', 
                                 total=len(qidxs) // batch_size,
                                 position=2, leave=False):
                X, _ = batch
                image_encoding = net.encoder(X.to(self.device))
                vlad_encoding = net.pool(image_encoding)
                qvecs[i * batch_size:(i + 1) * batch_size, :] = vlad_encoding
                del batch, X, image_encoding, vlad_encoding
            # release memory
            del qloader
            for i, batch in tqdm(enumerate(ploader_pc), 
                                 desc='compute positive descriptors', 
                                 total=len(pidxs) // batch_size,
                                 position=2, leave=False):
                X, _ = batch
                X = X.view((-1, 1, 4096, 3))
                vlad_encoding = net3d(X.to(self.device))
                pvecs[i * batch_size:(i + 1) * batch_size, :] = vlad_encoding
                del batch, X, vlad_encoding
            # release memory
            del ploader_pc
            for i, batch in tqdm(enumerate(nloader_pc), 
                                 desc='compute negative descriptors', 
                                 total=len(nidxs) // batch_size,
                                 position=2, leave=False):
                X, _ = batch
                X = X.view((-1, 1, 4096, 3))
                vlad_encoding = net3d(X.to(self.device))
                nvecs[i * batch_size:(i + 1) * batch_size, :] = vlad_encoding
                del batch, X, vlad_encoding
            # release memory
            del nloader_pc
        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)
        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        # the first return is the sorted tensor, the second return is the raw index that are sorted
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)
        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()
        # selection of hard triplets
        for q in range(len(qidxs)):
            qidx = qidxs[q]
            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))
            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))
            # take the closest positve
            dPos = pScores[q, pidx][0][0]
            # get distances to all negatives
            dNeg = nScores[q, :]
            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss
            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue
            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]
            # print('hardest_negIdx:----------------------------------')
            # print(hardest_negIdx)
            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]
            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]
            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]
            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]
            # if q < 20:
            #     print('triplet:----------------------------------')
            #     print(triplet)
            target = [-1, 1] + [0] * len(hardestNeg)
            self.triplets.append((triplet, target))
        # release memory
        del qvecs, nvecs, pScores, pRanks, nScores, nRanks
        # increment subset counter
        self.current_subset += 1

    @staticmethod
    def collate_fn(batch):
        """
        Creates mini-batch tensors from the list of tuples (query, positive, negatives).
        Args:
        batch: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None, None, None, None
        # zip single triplet to batches
        query, query_pc, positive, positive_pc, negatives_imgs, negatives_pcs, indices = zip(*batch)
        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        query_pc = data.dataloader.default_collate(query_pc)
        positive_pc = data.dataloader.default_collate(positive_pc)
        negatives_pcs = data.dataloader.default_collate(negatives_pcs)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives_imgs])
        negatives_imgs = torch.cat(negatives_imgs, 0)
        # negatives_imgs = data.dataloader.default_collate(negatives_imgs)
        # negatives_imgs = torch.cat(negatives_imgs, 0)
        # the query, positive, negatives indices are merged into list container
        indices = list(itertools.chain(*indices))
        return query, query_pc, positive, positive_pc, negatives_imgs, negatives_pcs, negCounts, indices

    def __getitem__(self, idx):
        '''
            for single triplet
            fetch query image, corresponding postives and negatives, idxes in current sequence data
        '''
        triplet, target = self.triplets[idx]
        # get query, positive and negative idx both for images and pcs
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]
        # load images and pcs into triplet list
        query = self.transform(Image.open(self.qImages[qidx]))
        positive_img = self.transform(Image.open(self.dbImages[pidx]))
        negatives_imgs = [self.transform(Image.open(self.dbImages[idx])) for idx in nidx]
        negatives_imgs = torch.stack(negatives_imgs, 0)
        query_pc = load_pc_files([self.qPcs[qidx]])
        positive_pc = load_pc_files([self.dbPcs[pidx]])
        negatives_pcs = load_pc_files([self.dbPcs[idx] for idx in nidx])
        return query, query_pc, positive_img, positive_pc, negatives_imgs, negatives_pcs, [qidx, pidx] + nidx
