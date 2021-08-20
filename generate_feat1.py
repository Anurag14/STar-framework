import numpy as np
import argparse
import shutil
import os
import torch
import torch.nn as nn
import tqdm
from pdb import set_trace as bkpt
from utils.return_dataset import return_dataset_test
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep, Predictor_feat
from collections import Counter
import math
import numpy as np

from torch.autograd import Variable
import scipy.spatial.distance
from math import ceil, floor
import shutil
import time
from operator import itemgetter
from scipy import stats
from model.resnet import resnet34, resnet50

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
#parser = argparse.ArgumentParser(description='SSDA Classification')
#arser.add_argument('--source', type=str, default='real',
#                   help='source domain')
#arser.add_argument('--target', type=str, default='sketch',
#                   help='target domain')
#arser.add_argument('--dataset', type=str, default='multi',
#                   choices=['multi', 'office', 'office_home', 'visda-2017'],
#                    help='the name of dataset')
#arser.add_argument('--num', type=int, default=3,
#                   help='number of labeled examples in the target')
#args = parser.parse_args()

def feature_val():

    source_all_feat = np.load('./inter_features/all_source_features.npy')
    source_all_gt = np.load('./inter_features/all_source_gt.npy')
    source_all_pred = np.load('./inter_features/all_source_pred.npy')

    target_feat = np.load('./inter_features/target_features.npy')
    target_labels = np.load('./inter_features/target_labels.npy')

    proto = np.load('./inter_features/prototypes.npy')
#bkpt()
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('features stats --------------------')
    print('Source all features: '+str(source_all_feat.shape))
    print('Source all samples predictions: '+str(source_all_pred.shape))
    print('Source all samples ground-truth labels: '+str(source_all_gt.shape))

    print('Target features: '+str(target_feat.shape))
    print('Target ground-truth labels: '+str(target_labels.shape))

    print('Class prototypes: '+ str(proto.shape))

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    return source_all_feat, source_all_gt, source_all_pred, target_feat, target_labels, proto

#def accuracy_pseudo(target_labels, select_t_pred, select_t_gt, select_t_id, args):
def accuracy_pseudo(target_labels, select_t_gt, select_t_id, args):
    overall_acc = 0    
    unique_target_labels=np.unique(target_labels)
    base_path = './data/txt/%s' % args.dataset
    image_set_file_t = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    with open(image_set_file_t, "r") as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
        image_index = itemgetter(*select_t_id)(image_index)

    if args.setting == 'SSDA':
        name_file = 'pseudo_images_' + args.source + '_' + args.target + '_%d.txt' % (args.num)
    else:
        name_file = 'pseudo_images_' + args.source + '_' + args.target + '.txt'
    file_name = open(name_file, "a")

    for ind, path in enumerate(image_index):
            file_name.write("%s %d\n" % (path, select_t_gt[ind]))
    file_name.close()
    shutil.copy(name_file, base_path)

def accuracy_pseudo1(target_labels, select_t_gt, select_t_id, args):
    overall_acc = 0
    unique_target_labels=np.unique(target_labels)
    base_path = './data/txt/%s' % args.dataset
    image_set_file_t = os.path.join(base_path, 'pseudo_images_' + args.source + '_' + args.target + '.txt')

    with open(image_set_file_t, "r") as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
        image_index = itemgetter(*select_t_id)(image_index)
    if args.setting == 'SSDA':
        name_file = 'mean_select_images_' + args.source + '_' + args.target + '_%d.txt' % (args.num)
    else:
        name_file = 'mean_select_images_' + args.source + '_' + args.target + '.txt'
    file_name = open(name_file, "a")
    for ind, path in enumerate(image_index):
        file_name.write("%s %d\n" % (path, select_t_gt[ind]))
    file_name.close()
    shutil.copy(name_file, base_path)

def accuracy_pseudo2(target_labels, select_t_gt, select_t_id, args):
    overall_acc = 0
    unique_target_labels=np.unique(target_labels)
    base_path = './data/txt/%s' % args.dataset
    image_set_file_t = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    with open(image_set_file_t, "r") as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
        image_index = itemgetter(*select_t_id)(image_index)
    if args.setting == 'SSDA':
        name_file = 'mean_select_images_' + args.source + '_' + args.target + '_%d.txt' % (args.num)
    else:
        name_file = 'mean_select_images_' + args.source + '_' + args.target + '.txt'
    file_name = open(name_file, "a")
    for ind, path in enumerate(image_index):
        file_name.write("%s %d\n" % (path, select_t_gt[ind]))

    file_name.close()
    shutil.copy(name_file, base_path)

def evaluate(args, ste, output_file="output.txt"):
    
    if args.net == 'resnet34':
        G = resnet34()
        inc = 512
    elif args.net == 'resnet50':
        G = resnet50()
        inc = 2048
    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
    elif args.net == "vgg":
        G = VGGBase()
        inc = 4096
    else:
        raise ValueError('Model cannot be recognized.')

    target_loader_unl_feat, source_loader_feat, class_list = return_dataset_test(args)
    if "resnet" in args.net:
        F1 = Predictor_feat(num_class=len(class_list),
                        inc=inc)
    else:
        F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
    G.cuda()
    F1.cuda()

    bkpt()
    ste = ste-1
    G.load_state_dict(torch.load(os.path.join(args.checkpath,
                                          "G_iter_model_{}_{}_"
                                          "to_{}_step_{}.pth.tar".
                                          format(args.method, args.source,
                                                 args.target, ste))))
    F1.load_state_dict(torch.load(os.path.join(args.checkpath,
                                           "F1_iter_model_{}_{}_"
                                           "to_{}_step_{}.pth.tar".
                                           format(args.method, args.source,
                                                  args.target, ste))))
    if "resnet" in args.net:
        protos = F1.fc2.weight.data.cpu().numpy()
    else:
        protos = F1.fc.weight.data.cpu().numpy()
    np.save('./inter_features/prototypes', protos, allow_pickle=True, fix_imports=True)

    
    size = 0
    im_data_t_l = torch.FloatTensor(1)
    gt_labels_t_l = torch.LongTensor(1)

# for source sample check
    im_data_s_l = torch.FloatTensor(1)
    gt_labels_s_l = torch.LongTensor(1)
   # gt_labels_s1 = torch.LongTensor(1)

    im_data_t_l = im_data_t_l.cuda()
    gt_labels_t_l = gt_labels_t_l.cuda()

# for source sample check
    im_data_s_l = im_data_s_l.cuda()
    gt_labels_s_l = gt_labels_s_l.cuda()

# for source sample check
    im_data_s_l = Variable(im_data_s_l)
    gt_labels_s_l = Variable(gt_labels_s_l)

    im_data_t_l = Variable(im_data_t_l)
    gt_labels_t_l = Variable(gt_labels_t_l)

    target_features = []
    target_labels = []
    query_features = []
    query_labels = []
    query_labels_pred = []
    correct = 0
    G.eval()
    F1.eval()
    bkpt()
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(target_loader_unl_feat)):
                im_data_t_l.data.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t_l.data.resize_(data_t[1].size()).copy_(data_t[1])
               # bkpt()
               # paths = data_t[2]
                feat = G(im_data_t_l)
                _, feats = F1(feat)
                target_features.append(feats)
                target_labels.extend(gt_labels_t_l.data.cpu().numpy())
           # bkpt()
            features = torch.cat(target_features)
            labels = np.asarray(target_labels)
            np.save('./inter_features/target_features', features, allow_pickle=True, fix_imports=True)
            np.save('./inter_features/target_labels', labels, allow_pickle=True, fix_imports=True)

        #with torch.no_grad():
            for batch_idx1, data_s in tqdm(enumerate(source_loader_feat)):
                im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
                gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
#                gt_labels_s1.data.resize_(data_s[2].size()).copy_(data_s[2])
              #  paths = data_t[2]
                feat = G(im_data_s)
                output1, feats = F1(feat)
                query_features.append(feats)
                query_labels.extend(gt_labels_s.data.cpu().numpy())
                pred1 = output1.data.max(1)[1]
#                query_labels_correct.extend(gt_labels_s1.data.cpu().numpy())
                query_labels_pred.extend(pred1.data.cpu().numpy())
            query_features = torch.cat(query_features)
            query_labels = np.asarray(query_labels)
            query_labels_pred = np.asarray(query_labels_pred)
            np.save('./inter_features/all_source_features', query_features, allow_pickle=True, fix_imports=True)
            np.save('./inter_features/all_source_gt', query_labels, allow_pickle=True, fix_imports=True)
            np.save('./inter_features/all_source_pred', query_labels_pred, allow_pickle=True, fix_imports=True)

def means_target(s_feat, s_label, s_pred, proto, t_feat, t_label, p_count, idz, sort_dist, args):
    num_t, feat_dim = t_feat.shape
    if args.dataset == 'multi':
        num_class = 126
    elif args.dataset == 'office':
        num_class = 31
    else:
        num_class = 65

    target_index = []
    Kmean = KMeans(n_clusters = num_class*2)
    Kmean.fit(t_feat)
    ctrd = Kmean.cluster_centers_

    for idx in range(0, num_class*2):

        class_proto = ctrd[idx,:].reshape((1,feat_dim))
        dist_ = scipy.spatial.distance.cdist(class_proto, t_feat,'euclidean')
        sort_dist_idx = np.argsort(dist_, axis=1) 
        t_index = sort_dist_idx[:, 0]
       # target_index = np.append(target_index, t_index, axis=0)
        target_index.append(t_index)
   # print(target_index.shape)
    inter_t_feat = t_feat[target_index,:]      
    inter_t_feat = inter_t_feat.squeeze()
    inter_t_label = t_label[target_index]
    inter_t_label = inter_t_label.squeeze()

    dist_ =  scipy.spatial.distance.cdist(inter_t_feat, proto, metric='euclidean')  
    ids = np.argmin(dist_, axis=1)
    dist = np.min(dist_, axis=1)
  #  min_dist = np.min(dist, ais=0)
    for i, di in enumerate(dist):
        if ids[i] in idz:
            index = np.where(ids[i]==idz)
           # dist[i] = dist[i]/sort_dist[index]
            dist[i] = dist[i]*sort_dist[index]
           # dist[i] = dist[i]/0.8
        else:
            pass

    #dist_ = np.min(dist_, axis=1)
    target_ids = np.argsort(dist, axis=0)
    target_ids = target_ids[num_class:]
    target_index = np.asarray(target_index)
    target_index = target_index[target_ids]
    target_index = target_index.squeeze()
    return target_index

def _means_target(s_feat, s_label, s_pred, proto, t_feat, t_label, p_count, args):
    num_t, feat_dim = t_feat.shape
    target_index = []

    if args.dataset == 'multi':
        num_class = 126
    elif args.dataset == 'office':
        num_class = 31
    else:
        num_class = 65

    Kmean = KMeans(n_clusters = num_class)
    Kmean.fit(t_feat)
    ctrd = Kmean.cluster_centers_
    for idx in range(0, num_class): 

        class_proto = ctrd[idx,:].reshape((1,feat_dim))
        dist_ = scipy.spatial.distance.cdist(class_proto, t_feat,'euclidean')
        sort_dist_idx = np.argsort(dist_, axis=1) 
        t_index = sort_dist_idx[:, 0]
        target_index.append(t_index)
       #target_index = np.append(target_index, t_index, axis=0)
    target_index = np.asarray(target_index)
    target_index = target_index.squeeze()
    return target_index

def update_distances(cluster_centers, already_selected, min_distances, t_feat, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.
    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = (t_feat[cluster_centers, :]).reshape(1, t_feat.shape[1])
      dist = pairwise_distances(t_feat, x, metric='euclidean')

      if min_distances is None:
        min_distances = np.min(dist, axis=1).reshape(-1,1)
        return min_distances
      else:
        min_distances = np.minimum(min_distances, dist)
        return min_distances

def coreset(t_feat, t_label, p_count, args):
    n_obs, _ = t_feat.shape
    new_batch = []
    already_select = []
    N = 126
    min_distances = None
    for _ in range(N):
      if len(already_select) == 0:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(n_obs))
      else:
        ind = np.argmax(min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_select

      min_distances = update_distances([ind], already_select, min_distances, t_feat, only_new=True, reset_dist=False)
      new_batch.append(ind)
      already_select.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(min_distances))

 #   already_selected = already_selected
    new_batch = np.asarray(new_batch)
    return new_batch

def select_target(s_feat, s_label, s_pred, proto, t_feat, t_label, p_count, args):
    
    unique_s_label = np.unique(s_label)
    num_s, feat_dim = s_feat.shape
    num_t = t_feat.shape[0]
    
    s_class_rep = np.empty((0,feat_dim))
    s_class_rep_gt = np.empty((0,))
    s_class_max_spread = np.empty((0,))
    s_class_min_spread = np.empty((0,))
    unique_label, count = np.unique(s_label, return_counts=True)
    num_class = len(unique_label)
    
    select_target_idx = np.empty((0,))
    select_target_pred = np.empty((0,))

    dist_proto_t = scipy.spatial.distance.cdist(proto,t_feat,'euclidean')
    max_K_t=1
    
    for idx in range(0,num_class):
        
        class_ = unique_label[idx]
        count_per_class = count[idx]
        s_label_idx = np.where(np.equal(s_label, class_ ))[0]
        s_pred_idx = np.where(np.equal(s_pred, class_))[0]
        s_correct_pred_idx = np.intersect1d(s_label_idx,s_pred_idx)
        
        s_correct_pred_feat = s_feat[s_correct_pred_idx,:]
        s_correct_pred_gt = s_label[s_correct_pred_idx]
        count_class = len(s_correct_pred_idx)
        class_proto = proto[idx,:].reshape((1,feat_dim))
        dist_ = scipy.spatial.distance.cdist(class_proto, s_correct_pred_feat,'euclidean')
        sorted_dist_idx = np.argsort(dist_,axis=1)
        #max_K_distant_feat_idx = sorted_dist_idx[:,-max_K:].reshape((max_K,))
       
        ctr = 0
        if p_count == 1:
            max_K_distant_feat_idx = sorted_dist_idx[:,:(ceil(count_class/4))].squeeze()
            ctr = 1
        #    max_K_distant_feat_idx = sorted_dist_idx[:,-(ceil(count_class/2)):-(ceil(count_class/4))].squeeze()
        elif p_count == 2:
#            max_K_distant_feat_idx = sorted_dist_idx[:,-(ceil(3*count_per_class/4)):-(ceil(count_per_class/2))].squeeze()
            max_K_distant_feat_idx = sorted_dist_idx[:,ceil(count_class/4):ceil(count_class/2)].squeeze()
            ctr = 2
        elif p_count == 3:
            max_K_distant_feat_idx = sorted_dist_idx[:,-(ceil(count_class/2)):-(ceil(count_class/4))].squeeze()
            ctr = 3
#        max_K_distant_feat_idx = sorted_dist_idx[:,-(ceil(count_per_class/2)):-(ceil(count_per_class/4))].squeeze()
        if max_K_distant_feat_idx.size == 0:
            continue
        s_maxK_dis_feat = s_correct_pred_feat[max_K_distant_feat_idx,:]
        s_maxK_dis_label = s_correct_pred_gt[max_K_distant_feat_idx]
    #    bkpt()
        s_max_spread_class = dist_[:,max_K_distant_feat_idx[-1]]
        s_min_spread_class = dist_[:,max_K_distant_feat_idx[0]]
        
        # K-source samples at max distance from class proto, but correctly classified
        # features, ground-truth labels and maximum spread of the class from the prototype are strored
#        s_class_rep = np.append(s_class_rep,s_maxK_dis_feat,axis=0)  
#        s_class_rep_gt = np.append(s_class_rep_gt,s_maxK_dis_label,axis=0)
#        s_class_max_spread = np.append(s_class_max_spread,s_max_spread_class,axis=0)
#        s_class_min_spread = np.append(s_class_min_spread,s_min_spread_class,axis=0)
        
    
  ## dist_proto_t = scipy.spatial.distance.cdist(proto,t_feat,'euclidean')
  ## max_K_t=3
   # for idx in range(0,num_class):
        
   #     class_=unique_label[idx]
        all_t_dist = dist_proto_t[idx,:]
#        same_class_radius_t_idx = np.where(all_t_dist < s_class_max_spread[idx])[0]# add [idx] when you loop
        same_class_radius_t_idx = np.where(all_t_dist < s_max_spread_class)[0]       
        if same_class_radius_t_idx.size == 0:
#            continue 
            if ctr == 1:
                max_K_distant_feat_idx = sorted_dist_idx[:,ceil(count_class/4):ceil(count_class/2)].squeeze()            
            elif ctr == 2:
               max_K_distant_feat_idx = sorted_dist_idx[:,-(ceil(count_class/2)):-(ceil(count_class/4))].squeeze()
            elif ctr == 3:
               max_K_distant_feat_idx = sorted_dist_idx[:,-(ceil(count_class/4)):].squeeze()
       #     else:
       
            s_maxK_dis_feat = s_correct_pred_feat[max_K_distant_feat_idx,:]
            s_maxK_dis_label = s_correct_pred_gt[max_K_distant_feat_idx]
     #       bkpt()
            s_max_spread_class = dist_[:,max_K_distant_feat_idx[-1]]
            s_min_spread_class = dist_[:,max_K_distant_feat_idx[0]]
            same_class_radius_t_idx = np.where(all_t_dist < s_max_spread_class)[0]
            possible_class_idx = same_class_radius_t_idx
            possible_class_idx_dist = all_t_dist[possible_class_idx]
            sorted_possible_class_radius_t_idx = np.argsort(possible_class_idx_dist)
        else:

            possible_class_idx = same_class_radius_t_idx
            possible_class_idx_dist = all_t_dist[possible_class_idx]
            sorted_possible_class_radius_t_idx = np.flip(np.argsort(possible_class_idx_dist), axis=0)

        kdist_possible_class_radius_idx = possible_class_idx[sorted_possible_class_radius_t_idx]
        
        class_target_idx=np.empty((0,))
        if len(kdist_possible_class_radius_idx)!=0:
           # class_target_idx=np.empty((0,))
            for l in range(0,len(kdist_possible_class_radius_idx)):
                t_feat_l=t_feat[kdist_possible_class_radius_idx[l],:].reshape((1,feat_dim))
                dist_=scipy.spatial.distance.cdist(t_feat_l,proto,'euclidean')
                sort_dist_idx = np.argsort(dist_,axis=1)
                nn_proto_idx = sort_dist_idx[:,0]
                if nn_proto_idx==class_:
                    class_target_idx = np.append(class_target_idx,[kdist_possible_class_radius_idx[l]],axis=0)
                if len(class_target_idx)==max_K_t:
                    break
        
        select_target_idx = np.append(select_target_idx,class_target_idx,axis=0)
        select_target_pred = np.append(select_target_pred,np.tile(class_,len(class_target_idx)),axis=0)
        
    return select_target_idx, select_target_pred
