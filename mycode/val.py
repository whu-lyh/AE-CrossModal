'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

Validation of NetVLAD, using the Mapillary Street-level Sequences Dataset.
'''


import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from mycode.msls import ImagesFromList
from mycode.msls import PcFromFiles
from crossmodal.tools.datasets import input_transform


def val(eval_set, model, model3d, encoder_dim, device, threads, config, writer, size, epoch_num=0, write_tboard=False, pbar_position=0):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform(size,train=False))
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform(size,train=False))
    eval_set_queries_pc = PcFromFiles(eval_set.qPcs)
    eval_set_dbs_pc = PcFromFiles(eval_set.dbPcs)
    test_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                          num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                          shuffle=False, pin_memory=cuda)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                         num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                         shuffle=False, pin_memory=cuda)
    test_data_loader_queries_pc = DataLoader(dataset=eval_set_queries_pc,
                                         num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                         shuffle=False, pin_memory=cuda)
    test_data_loader_dbs_pc = DataLoader(dataset=eval_set_dbs_pc,
                                      num_workers=threads, batch_size=int(config['train']['cachebatchsize']),
                                      shuffle=False, pin_memory=cuda)

    model.eval()
    model3d.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        '''pool_size = encoder_dim
        if config['global_params']['pooling'].lower() == 'netvlad':
            pool_size *= int(config['global_params']['num_clusters'])'''
        pool_size = 256
        qFeat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        dbFeat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)
        qFeat_pc = np.empty((len(eval_set_queries_pc), pool_size), dtype=np.float32)
        dbFeat_pc = np.empty((len(eval_set_dbs_pc), pool_size), dtype=np.float32)

        for feat, test_data_loader in zip([qFeat, dbFeat], [test_data_loader_queries, test_data_loader_dbs]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test1 Iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                image_encoding = model.encoder(input_data)

                vlad_encoding = model.pool(image_encoding)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, image_encoding, vlad_encoding
        for feat, test_data_loader in zip([qFeat_pc, dbFeat_pc], [test_data_loader_queries_pc, test_data_loader_dbs_pc]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test2 Iter'.rjust(15)), 1):
                input_data = input_data.float()
                # feed_tensor = torch.cat((input_data), 1)
                input_data = input_data.view((-1, 1, 4096, 3))
                input_data = input_data.to(device)
                vlad_encoding = model3d(input_data)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, vlad_encoding


    del test_data_loader_queries, test_data_loader_dbs, test_data_loader_queries_pc, test_data_loader_dbs_pc

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat_pc)

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50]

    # for each query get those within threshold distance
    gt = eval_set.all_pos_indices
    # print('ground_truth')
    # print(gt)

    # any combination of mapillary cities will work as a val set
    # 2d->3d
    predictions = {}
    predictions_t = {}
    des = ['2d->2d', '2d->3d', '3d->2d', '3d->3d']
    for i in range(4):
        if i == 0:   # 2d->2d
            qTest = qFeat
            dbTest = dbFeat
        if i == 1:   # 2d->3d
            qTest = qFeat
            dbTest = dbFeat_pc
        if i == 2:   # 3d->2d
            qTest = qFeat_pc
            dbTest = dbFeat
        if i == 3:   # 3d->3d
            qTest = qFeat_pc
            dbTest = dbFeat_pc
        qEndPosTot = 0
        dbEndPosTot = 0
        for cityNum, (qEndPos, dbEndPos) in enumerate(zip(eval_set.qEndPosList, eval_set.dbEndPosList)):
            faiss_index = faiss.IndexFlatL2(pool_size)
            faiss_index.add(dbTest[dbEndPosTot:dbEndPosTot+dbEndPos, :])
            _, preds = faiss_index.search(qTest[qEndPosTot:qEndPosTot+qEndPos, :], max(n_values) + 1)
            if cityNum == 0:
                predictions_t[i] = preds
            else:
                predictions_t[i] = np.vstack((predictions_t, preds))
            qEndPosTot += qEndPos
            dbEndPosTot += dbEndPos
    # get rid of the same frame of query and database for same modality
    # predictions = predictions_t
    predictions[0] = [list(pre[0:]) for pre in predictions_t[0]]
    predictions[3] = [list(pre[0:]) for pre in predictions_t[3]]
    predictions[1] = [list(pre[:50]) for pre in predictions_t[1]]
    predictions[2] = [list(pre[:50]) for pre in predictions_t[2]]
    for i in range(4):
        print(des[i])
        print(predictions[i])
    recall_at_n = {}
    for test_index in range(4):
        correct_at_n = np.zeros(len(n_values))
        # TODO can we do this on the matrix in one go?
        for qIx, pred in enumerate(predictions[test_index]):
            for i, n in enumerate(n_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        recall_at_n[test_index] = correct_at_n / len(eval_set.qIdx)
    # 2d->2d
    for i, n in enumerate(n_values):
        tqdm.write("====> 2D->2D/Recall@{}: {:.4f}".format(n, recall_at_n[0][i]))
        if write_tboard:
            writer.add_scalar('2Dto2D/Recall@' + str(n), recall_at_n[0][i], epoch_num)
    # 2d->3d
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[1][i]
        tqdm.write("====> 2D->3D/Recall@{}: {:.4f}".format(n, recall_at_n[1][i]))
        if write_tboard:
            writer.add_scalar('2Dto3D/Recall@' + str(n), recall_at_n[1][i], epoch_num)
    # 3d->2d
    for i, n in enumerate(n_values):
        tqdm.write("====> 3D->2D/Recall@{}: {:.4f}".format(n, recall_at_n[2][i]))
        if write_tboard:
            writer.add_scalar('3Dto2D/Recall@' + str(n), recall_at_n[2][i], epoch_num)
    # 3d->3d
    for i, n in enumerate(n_values):
        tqdm.write("====> 3D->3D/Recall@{}: {:.4f}".format(n, recall_at_n[3][i]))
        if write_tboard:
            writer.add_scalar('3Dto3D/Recall@' + str(n), recall_at_n[3][i], epoch_num)

    return all_recalls
