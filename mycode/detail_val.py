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
import faiss.contrib.torch_utils
import os
import json
import shutil
from PIL import Image
from matplotlib import pyplot as plt
from prettytable import PrettyTable, MARKDOWN
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from mycode.msls import ImagesFromList
from mycode.msls import PcFromFiles
from crossmodal.tools.datasets import input_transform
from eval.visualization import tsne_visualization

def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=0).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


def save_recall_curve(k_values, topN_recall, result_path:str, dataname:str, model_name:str):
    fig, ax = plt.subplots(1, 1, dpi=300)
    plt.rcParams['font.size'] = '12'
    if (type(topN_recall).__name__ == 'dict'):
        plt.plot(topN_recall.keys(), topN_recall.values())
        # assign x axis markers based on dict, which indicates the k value
        recall_list = []
        for k in topN_recall:
            recall_list.append(k)
    else:
        plt.plot([k for k in k_values], topN_recall)  
    plt.xticks([k for k in k_values])
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel('Top @N')
    plt.ylabel('Recall (%)')
    plt.title('Recall on sequence {} from task {}'.format(dataname, model_name))
    fig_name = 'Top_N_dataset{}_{}.png'.format(dataname, model_name)
    plt.savefig(os.path.join(result_path, fig_name))


def save_recall_table(k_values, recall_at_k, task_name=""):
    table = PrettyTable()
    table.field_names = ['K']+[str(k) for k in k_values]
    table.add_row(['Recall@K']+ [f'{v:.2f}' for v in recall_at_k])
    # markdown style results
    table.set_style(MARKDOWN)
    print(table.get_string(title=f"Performance on {task_name}"))


def compute_recall(gt, predictions, numQ, k_values, task_name=''):
    correct_at_k = np.zeros(len(k_values))
    for qIx, pred in enumerate(predictions):
        for i, k in enumerate(k_values):
            # if in top K then also in top NN, where NN > K
            if np.any(np.in1d(pred[:k], gt[qIx])):
                correct_at_k[i:] += 1
                break
    # len(predictions) equals to numQ
    recall_at_k = correct_at_k / numQ * 100
    for i, k in enumerate(k_values):
        tqdm.write("====> {} Recall@{}: {:.2f}".format(task_name, k, recall_at_k[i]))
    return recall_at_k


def val(eval_set, model2d, model3d, threads, batch_size:int, result_path, faiss_gpu=True, save_fig=True, pbar_position=0, debug=True):
    '''
        Validation function offline, more close to test on single sequence
    '''
    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform(train=False))
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform(train=False))
    eval_set_queries_pc = PcFromFiles(eval_set.qPcs)
    eval_set_dbs_pc = PcFromFiles(eval_set.dbPcs)
     # shuffle here is used to get feature fulfulled, and no affect to the final prediction result order
    test_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                    num_workers=threads, batch_size=batch_size,
                                    shuffle=False, pin_memory=True)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                num_workers=threads, batch_size=batch_size,
                                shuffle=False, pin_memory=True)
    test_data_loader_queries_pc = DataLoader(dataset=eval_set_queries_pc,
                                        num_workers=threads, batch_size=batch_size,
                                        shuffle=False, pin_memory=True)
    test_data_loader_dbs_pc = DataLoader(dataset=eval_set_dbs_pc,
                                    num_workers=threads, batch_size=batch_size,
                                    shuffle=False, pin_memory=True)
    # for each query get those within threshold distance, and save GT into json
    gt = eval_set.all_pos_indices
    gt_dists = eval_set.all_pos_dists
    gt_index = []
    gt_lists = []
    for i in range(len(gt)):
        gt_index.append(os.path.basename(eval_set.qImages[i]))
        pos_indices = gt[i]
        pics = [os.path.basename(eval_set.dbImages[p]) for p in pos_indices]
        gt_lists.append(pics)
    gt_dic = dict(zip(gt_index, gt_lists))
    with open(os.path.join(result_path, "ground_truth.json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(gt_dic, indent=4))
    # save corresponding distances for further check
    if debug:
        gt_dist_lists = []
        for i in range(len(gt_dists)):
            pos_indices = gt_dists[i]
            pics = [p for p in pos_indices]
            gt_dist_lists.append(pics)
        gt_dist_dic = dict(zip(gt_index, gt_dist_lists))
        with open(os.path.join(result_path, "ground_truth_dists.json"), 'w', encoding='utf-8') as file:
            file.write(json.dumps(gt_dist_dic, indent=4))
    k_values = [1, 5, 10, 20, 50]
    des = ['2dto2d', '2dto3d', '3dto2d', '3dto3d']
    # top1_percent = int(len(eval_set_queries)) * 0.01 + 1
    # k_values.append(top1_percent)
    predictions_tmp = {}
    predictions_dists_tmp = {} # no further usage now
    eval_set_query_num = len(eval_set_queries)
    no_feature = False
    # if no model, then generate random results
    if model2d is None and model3d is None:
        for i in range(4):
            predictions_tmp[i] = np.empty((eval_set_query_num, max(k_values)), dtype=np.int)
            for query_idx in range(eval_set_query_num):
                predictions_tmp[i][query_idx] = np.random.choice(len(eval_set_dbs), max(k_values), replace=False)
        no_feature = True
    else:
        model2d.eval() # without calling this func, the performance drop heavily
        model3d.eval()
        with torch.no_grad():
            tqdm.write('====> Extracting Features')
            global_feature_dim = 256
            # image feature initialize
            qFeat = np.empty((eval_set_query_num, global_feature_dim), dtype=np.float32)
            qFeat_FeatureMap = np.empty((eval_set_query_num, 512, 32, 64), dtype=np.float32)
            dbFeat = np.empty((len(eval_set_dbs), global_feature_dim), dtype=np.float32)
            dbFeat_FeatureMap = np.empty((len(eval_set_dbs), 512, 32, 64), dtype=np.float32)
            # pc feature initialize
            qFeat_pc = np.empty((len(eval_set_queries_pc), global_feature_dim), dtype=np.float32)
            qFeat_pc_FeatureMap = np.empty((len(eval_set_queries_pc), 1024, 4096, 1), dtype=np.float32)
            dbFeat_pc = np.empty((len(eval_set_dbs_pc), global_feature_dim), dtype=np.float32)
            dbFeat_pc_FeatureMap = np.empty((len(eval_set_dbs_pc), 1024, 4096, 1), dtype=np.float32)
            # each for loop corresponding to a batch data
            for feat, feat_fm, test_data_loader in zip([qFeat, dbFeat], [qFeat_FeatureMap, dbFeat_FeatureMap], [test_data_loader_queries, test_data_loader_dbs]):
                for iteration, (input_data, indices) in \
                        enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Evaluate Images Iter'.rjust(15)), 1):
                    # print('input_data')
                    # print(input_data)
                    # print('input_data.shape')
                    # print(input_data.shape) # torch.Size([10, 3, 512, 1024])
                    # print('indices')
                    # print(indices)
                    input_data = input_data.to("cuda")
                    image_encoding = model2d.encoder(input_data)
                    vlad_encoding= model2d.pool(image_encoding)
                    # all image query and database will be calculated in a batch parallel manner
                    feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    # release memory
                    del input_data, image_encoding, vlad_encoding
            for feat, feat_fm, test_data_loader in zip([qFeat_pc, dbFeat_pc], [qFeat_pc_FeatureMap, dbFeat_pc_FeatureMap], [test_data_loader_queries_pc, test_data_loader_dbs_pc]):
                for iteration, (input_data, indices) in \
                        enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Evaluate Pcs Iter'.rjust(15)), 1):
                    # print('input_data3d')
                    # print(input_data)
                    # print('input_data3d.shape')
                    # print(input_data.shape) # torch.Size([10, 4096, 3])
                    # print('indices3d')
                    # print(indices)
                    input_data = input_data.float()
                    input_data = input_data.view((-1, 1, 4096, 3))
                    input_data = input_data.to("cuda")
                    # pc_enc = model3d.point_net(input_data)
                    # vlad_encoding = model3d.net_vlad(pc_enc)
                    vlad_encoding = model3d(input_data)
                    # all pc query and database will be calculated in a batch parallel manner
                    feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    # release memory
                    del input_data, vlad_encoding
        del test_data_loader_queries, test_data_loader_dbs, test_data_loader_queries_pc, test_data_loader_dbs_pc
        tqdm.write('====> Building faiss index')
        tqdm.write('====> Calculating recall @ N')
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
                # build index
                if faiss_gpu:
                    res = faiss.StandardGpuResources()
                    flat_config = faiss.GpuIndexFlatConfig()
                    flat_config.useFloat16 = True
                    flat_config.device = 0
                    faiss_index = faiss.GpuIndexFlatL2(res, global_feature_dim, flat_config)
                else:               
                    faiss_index = faiss.IndexFlatL2(global_feature_dim)
                # add the current sequence database data into faiss
                faiss_index.add(dbTest[dbEndPosTot:dbEndPosTot+dbEndPos, :])
                tsne_visualization(result_path, des[i] + "_query", qTest[qEndPosTot:qEndPosTot+qEndPos, :])
                tsne_visualization(result_path, des[i] + "_database", dbTest[dbEndPosTot:dbEndPosTot+dbEndPos, :])
                # search for each query data, could be done in a batch and return multi searching results by k_values(number check)
                # faiss will return indices and distances
                dis, preds = faiss_index.search(qTest[qEndPosTot:qEndPosTot+qEndPos, :], max(k_values) + 1) # add +1
                if cityNum == 0:
                    predictions_tmp[i] = preds
                    predictions_dists_tmp[i] = dis
                else:
                    predictions_tmp[i] = np.vstack((predictions_tmp[i], preds))
                    predictions_dists_tmp[i] = np.vstack((predictions_dists_tmp[i], dis))
                qEndPosTot += qEndPos
                dbEndPosTot += dbEndPos
    # fetch prediction results
    predictions = {}
    predictions[0] = [list(pre[:50]) for pre in predictions_tmp[0]] # 2d->2d
    predictions[1] = [list(pre[:50]) for pre in predictions_tmp[1]] # 2d->3d
    predictions[2] = [list(pre[:50]) for pre in predictions_tmp[2]] # 3d->2d
    predictions[3] = [list(pre[:50]) for pre in predictions_tmp[3]] # 3d->3d
    # predictions[0] = gt # 2d->2d
    # predictions[1] = gt # 2d->3d
    # predictions[2] = gt # 3d->2d
    # predictions[3] = gt # 3d->3d
    # sampled query data
    q_ind = []
    q_ind_imgs_path = []
    q_ind_pcs_path = []
    # db images or pcs
    pics_path = {}
    p_fnames = {}
    for i in range(4):
        pics_path[i] = []
        p_fnames[i] = []
    for i in range(len(gt)):
        # get query image name
        q_ind.append(os.path.basename(eval_set.qImages[i]))
        q_ind_imgs_path.append(eval_set.qImages[i])
        q_ind_pcs_path.append(eval_set.qPcs[i])
        # get predictions for each recall check
        for j in range(4):
            # prediction positive data indices
            pre_pos = predictions[j][i]
            # prediction data path only images
            pic_path = eval_set.dbImages[pre_pos]
            pics_path[j].append(pic_path)
            if j == 0 or j == 2:
                p_fname = [os.path.basename(eval_set.dbImages[p]) for p in pre_pos]
            else:
                p_fname = [os.path.basename(eval_set.dbPcs[p]) for p in pre_pos]
            p_fnames[j].append(p_fname)
    # save the whole prediction results into json file
    for i,task in enumerate(des):
        path = os.path.join(result_path, task + ".json")
        pre_dic = dict(zip(q_ind, p_fnames[i]))
        with open(path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(pre_dic, indent=4))
    # save all specific results
    if save_fig:
        # randomly select 10 query samples
        save_num = 10
        save_ind = np.random.choice(eval_set_query_num, save_num, replace=False)
        # indx_p = 0
        for indx_p, task_name in enumerate(des):
            task_path = os.path.join(result_path, task_name)
            if not os.path.exists(task_path):
                os.mkdir(task_path)
            # save results for each query data respectively
            for ind in save_ind:
                # here the id is the csv id, not the image id
                save_dir = os.path.join(task_path, str(ind))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                # copy query data(image and pc) to desired path
                save_q = os.path.join(save_dir, "query")
                if not os.path.exists(save_q):
                    os.mkdir(save_q)
                query_img = q_ind_imgs_path[ind]
                shutil.copy(query_img, save_q)
                query_pc = q_ind_pcs_path[ind]
                shutil.copy(query_pc, save_q)
                # save recalls
                save_recalls = os.path.join(save_dir, "recalls")
                if not os.path.exists(save_recalls):
                    os.mkdir(save_recalls)
                predict_imgs = pics_path[indx_p][ind][:5]
                #print("predict_imgs", predict_imgs)
                # for database data only file name is saved, no matter imgs or pcs
                predict_pcs = p_fnames[indx_p][ind][:5]
                if indx_p == 0 or indx_p == 2:
                    for i, pos in enumerate(predict_imgs):
                        shutil.copy(pos, os.path.join(save_recalls, os.path.basename(pos)))
                else:
                    for i, pos in enumerate(predict_pcs):
                        # here database data should and could not be zero
                        db_pc = os.path.join(os.path.dirname(eval_set.dbPcs[0]), pos)
                        shutil.copy(db_pc, os.path.join(save_recalls, pos))
                pos_index_indbs = predictions[indx_p][ind][:10]
                #print("pos_index_in_dbs:\t", pos_index_indbs)
                # save feature maps
                if not no_feature:
                    # feature map of query data
                    if indx_p == 0 or indx_p == 1:
                        enc = torch.from_numpy(qFeat_FeatureMap[ind])
                        vlad = torch.from_numpy(qFeat[ind])
                        save_img(enc.unsqueeze(0), save_q + '/encoding')
                        save_img(vlad.view(16, 16).unsqueeze(0).unsqueeze(0), save_q + '/vlad')
                    else:
                        enc = torch.from_numpy(qFeat_pc_FeatureMap[ind])
                        vlad = torch.from_numpy(qFeat_pc[ind])
                        save_img(enc.unsqueeze(0).view(1, 1024, 64, 64), save_q + '/encoding')
                        save_img(vlad.view(16, 16).unsqueeze(0).unsqueeze(0), save_q + '/vlad')
                    save_dir_fm = os.path.join(save_dir, "featureMaps")
                    if not os.path.exists(save_dir_fm):
                        os.mkdir(save_dir_fm)
                    if indx_p == 0 or indx_p == 2:
                        for i, ind_db in enumerate(pos_index_indbs):
                            enc_db = torch.from_numpy(dbFeat_FeatureMap[ind_db])
                            vlad_db = torch.from_numpy(dbFeat[ind_db])
                            save_enc_path = os.path.join(save_dir_fm, 'enc' + str(i))
                            save_vlad_path = os.path.join(save_dir_fm, 'vlad' + str(i))
                            save_img(enc_db.unsqueeze(0), save_enc_path)
                            save_img(vlad_db.view(16, 16).unsqueeze(0).unsqueeze(0), save_vlad_path)
                    else:
                        for i, ind_db in enumerate(pos_index_indbs):
                            enc_db = torch.from_numpy(dbFeat_pc_FeatureMap[ind_db])
                            vlad_db = torch.from_numpy(dbFeat_pc[ind_db])
                            save_enc_path = os.path.join(save_dir_fm, 'enc' + str(i))
                            save_vlad_path = os.path.join(save_dir_fm, 'vlad' + str(i))
                            save_img(enc_db.unsqueeze(0).view(1, 1024, 64, 64), save_enc_path)
                            save_img(vlad_db.view(16, 16).unsqueeze(0).unsqueeze(0), save_vlad_path)
    # save recalls and directly save pr curves
    for task_index, task in enumerate(des):
        recall_at_k = compute_recall(gt=gt, predictions=predictions[task_index], 
                                    numQ=len(eval_set.qIdx), k_values=k_values, task_name=task)
        
        save_recall_curve(k_values, recall_at_k, result_path, dataname=eval_set.cities[0], model_name=task)
        save_recall_table(k_values, recall_at_k, task_name=task)
        print('\n')