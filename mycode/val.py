
import numpy as np
import faiss
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from mycode.msls import ImagesFromList
from mycode.msls import PcFromFiles
from crossmodal.tools.datasets import input_transform


def val(eval_dataset, model2d, model3d, encoder_dim, device, threads, config, writer, 
        epoch_num=0, write_tboard=True, pbar_position=0):
    '''
        Validation function while training
    '''
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
    # fetch validation datasets
    eval_dataset_queries = ImagesFromList(eval_dataset.qImages, transform=input_transform(train=False))
    eval_dataset_dbs = ImagesFromList(eval_dataset.dbImages, transform=input_transform(train=False))
    eval_dataset_queries_pc = PcFromFiles(eval_dataset.qPcs)
    eval_dataset_dbs_pc = PcFromFiles(eval_dataset.dbPcs)
    # dataloader
    test_data_loader_queries = DataLoader(dataset=eval_dataset_queries,
                                            num_workers=threads, 
                                            batch_size=int(config['train']['cachebatchsize']),
                                            shuffle=False, pin_memory=cuda)
    test_data_loader_dbs = DataLoader(dataset=eval_dataset_dbs,
                                        num_workers=threads, 
                                        batch_size=int(config['train']['cachebatchsize']),
                                        shuffle=False, pin_memory=cuda)
    test_data_loader_queries_pc = DataLoader(dataset=eval_dataset_queries_pc,
                                                num_workers=threads, 
                                                batch_size=int(config['train']['cachebatchsize']),
                                                shuffle=False, pin_memory=cuda)
    test_data_loader_dbs_pc = DataLoader(dataset=eval_dataset_dbs_pc,
                                            num_workers=threads, 
                                            batch_size=int(config['train']['cachebatchsize']),
                                            shuffle=False, pin_memory=cuda)
    # model freeze BN and dropout while validation
    model2d.eval()
    model3d.eval()
    # without gradient and save GPU memory
    with torch.no_grad():
        tqdm.write('====> Extracting Features for query and database images and pcs')
        # initialize empty containers for all query data and database datas, which could be memory consuming
        global_feature_dim = 256
        qFeat = np.empty((len(eval_dataset_queries), global_feature_dim), dtype=np.float32)
        dbFeat = np.empty((len(eval_dataset_dbs), global_feature_dim), dtype=np.float32)
        qFeat_pc = np.empty((len(eval_dataset_queries_pc), global_feature_dim), dtype=np.float32)
        dbFeat_pc = np.empty((len(eval_dataset_dbs_pc), global_feature_dim), dtype=np.float32)
        # get images' feature
        for feat, test_data_loader in zip([qFeat, dbFeat], [test_data_loader_queries, test_data_loader_dbs]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Query(image) feature generation iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                image_encoding = model2d.encoder(input_data)
                vlad_encoding = model2d.pool(image_encoding)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                # release memory
                del input_data, image_encoding, vlad_encoding
        # get pcs' feature
        for feat, test_data_loader in zip([qFeat_pc, dbFeat_pc], [test_data_loader_queries_pc, test_data_loader_dbs_pc]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Database(pc) feature generation iter'.rjust(15)), 1):
                input_data = input_data.float()
                input_data = input_data.view((-1, 1, 4096, 3))
                input_data = input_data.to(device)
                vlad_encoding = model3d(input_data)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                # release memory
                del input_data, vlad_encoding
    # release memory
    del test_data_loader_queries, test_data_loader_dbs, test_data_loader_queries_pc, test_data_loader_dbs_pc
    # build index
    tqdm.write('====> Building faiss index for database')
    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50]
    # for each query get those within threshold distance
    # The gt indices are found by knn from database images
    gt = eval_dataset.all_pos_indices
    # knn searching
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
        # qEndPosList storing the query data indices, while dbEndPosList for database data indices
        # all data are loaded from predifined csv files
        for cityNum, (qEndPos, dbEndPos) in enumerate(zip(eval_dataset.qEndPosList, eval_dataset.dbEndPosList)):
            faiss_index = faiss.IndexFlatL2(global_feature_dim)
            # add specific data indices in database for searching, due to all sequences used for test are sotred in one list
            # if there is only one test sequence, this operation is useless, but for multi-sequences!!!
            faiss_index.add(dbTest[dbEndPosTot:dbEndPosTot+dbEndPos, :])
            # search for each query data, could be done in a batch and return multi searching results by n_values
            # faiss will return indices and distances
            _, preds = faiss_index.search(qTest[qEndPosTot:qEndPosTot+qEndPos, :], max(n_values) + 1)
            # stack the results for all test sequences
            if cityNum == 0:
                predictions_t[i] = preds
            else:
                predictions_t[i] = np.vstack((predictions_t, preds))
            # move to next test indices intervals
            qEndPosTot += qEndPos
            dbEndPosTot += dbEndPos
    # fetch the first 50 prediction results
    predictions[0] = [list(pre[:50]) for pre in predictions_t[0]] # 2d->2d
    predictions[1] = [list(pre[:50]) for pre in predictions_t[1]] # 2d->3d
    predictions[2] = [list(pre[:50]) for pre in predictions_t[2]] # 3d->2d
    predictions[3] = [list(pre[:50]) for pre in predictions_t[3]] # 3d->3d
    # result check compared with GT files
    recall_at_n = {}
    for test_index in range(4):
        correct_at_n = np.zeros(len(n_values))
        for qIx, pred in enumerate(predictions[test_index]):
            for i, n in enumerate(n_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        # len(eval_dataset.qIdx) equals to predictions[test_index]
        recall_at_n[test_index] = correct_at_n / len(eval_dataset.qIdx)
    # 2d->2d
    for i, n in enumerate(n_values):
        tqdm.write("====> 2D->2D/Recall@{}: {:.4f}".format(n, recall_at_n[0][i]))
        if write_tboard:
            writer.add_scalar('2Dto2D/Recall@' + str(n), recall_at_n[0][i], epoch_num)
    # 2d->3d results will be returned
    all_recalls = {}
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
    # return 2d-3D recalls
    return all_recalls
