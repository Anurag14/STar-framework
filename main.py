from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from model.cbfocal import CB_loss
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset, return_dataset_test, return_pseudo, return_pseudo1
from utils.return_dataset import return_labeled_data, return_eval_data, return_test_data
from utils.return_indices import return_index
from utils.data_augmentation import data_augmentation_reg
from utils.loss import entropy, adentropy
from pdb import set_trace as bkpt
from generate_feat1 import *
from generate_feat1 import _means_target
import torch.utils.data as data1
import torch.utils.data.sampler  as sampler
from operator import itemgetter
from scipy import stats
from collections import Counter
import scipy.spatial.distance
from math import ceil, floor
import shutil
import time
from tqdm import tqdm
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=32001, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./_offactfinal_save_ssda_model',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--num_shot', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=10,metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--update_pseudo', type=int, default=1501, metavar='P',
                    help='when to update the pseudo labels ')
parser.add_argument('--start_weighted_entropy', type=int, default=10000, metavar='P',
                            help='when to update the pseudo labels ')
parser.add_argument('--update_dist', type=int, default=200, metavar='P',
                    help='when to update the distances metric ')
parser.add_argument('--setting', type=str, default='UDA',
                    choices=['UDA', 'SSDA'],
                    help='select UDA or SSDA for method setting')
args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

source_loader, target_loader_unl, class_list, cls_num_list, t_length = return_dataset(args)
target_loader_unl_feat, source_loader_feat, class_list = return_dataset_test(args)

num_images = return_index(args)
initial_indices = np.arange(num_images)

use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if args.net == 'alexnet':
    bs = 32
else:
    bs = 24
if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)


im_data_t_lab = torch.FloatTensor(1)
gt_labels_t_lab = torch.LongTensor(1)
im_data_t_lab = im_data_t_lab.cuda()
gt_labels_t_lab = gt_labels_t_lab.cuda()
im_data_t_lab = Variable(im_data_t_lab)
gt_labels_t_lab = Variable(gt_labels_t_lab)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()


im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

shot = args.num_shot
checkpath = args.checkpath + '_' + args.source + '_' + args.target +'_' +str(shot) + 'shot_' + args.net

if os.path.exists(checkpath) == False:
    os.mkdir(checkpath)

def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    #criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0
    p_update_count = 0
    for ste in range(all_step):
        if 0 <= (ste%args.update_pseudo)< args.update_pseudo:
            if ste%args.update_pseudo == int(t_length/(8*bs)):
                if "resnet" in args.net:
                    protos = F1.fc2.weight.data.cpu().numpy()
                else:
                    protos = F1.fc.weight.data.cpu().numpy()
                print('original protos shape:'+str(protos.shape))
                np.save('./inter_features/protos', protos, allow_pickle=True, fix_imports=True)
                distance = []
                epoch = []
                dist_arr = np.empty((0, 1))
                z = 0
            if ste%(int(t_length/(bs*4))) == 0 and p_update_count<args.num_shot and ste>0:
                z+=1
                protos = np.load('./inter_features/protos.npy')
                current = get_proto(target_loader_unl)
                alpha = 0.8
                distances = pairwise_distances(protos, current)
                np.save('./inter_features/protos', current, allow_pickle=True, fix_imports=True)
                dist_ = distances.diagonal()
                distance.append(dist_)
                if z ==1:
                    dist = dist_
                else:
                    dist = (1-alpha)*dist + alpha*dist_

        if ste % args.update_pseudo==0  and ste>0 and p_update_count<args.num_shot:

            dist_arr = np.asarray(dist)
            dist_mean = np.mean(dist_arr)
            idx = np.where(dist_arr>dist_mean)[0]
            sort_dist = dist_arr[idx]
            sort_dist = sort_dist/dist_mean            
            p_update_count+=1
            if p_update_count == 1:
                eval_(target_loader_unl_feat, source_loader_feat) 
            else:
                eval_sampler = data1.SequentialSampler(lab_t_index)
                target_loader_eval = return_eval_data(args, eval_sampler)
                eval_(target_loader_eval, source_loader_feat)

            source_all_feat, source_all_gt, source_all_pred, target_feat, target_labels, proto = feature_val()

            if p_update_count == 1:
                target_index = means_target(source_all_feat, source_all_gt, source_all_pred, proto, target_feat, target_labels, p_update_count, idx, sort_dist, args)
                picked_indices = list(target_index)
            else:
                target_index = means_target(source_all_feat, source_all_gt, source_all_pred, proto, target_feat, target_labels, p_update_count, idx, sort_dist, args)
                target_index = target_index.astype(int)
                target_index = lab_t_index[target_index]
                picked_indices = list(picked_indices) + list(target_index)
            lab_t_index = np.setdiff1d(list(initial_indices), picked_indices)
            sampler = data1.SubsetRandomSampler(lab_t_index)
            test_sampler = data1.SequentialSampler(lab_t_index)
            tar_sampler = data1.SubsetRandomSampler(picked_indices)
            target_loader_lab, target_loader_unl_1 = return_labeled_data(args, sampler, tar_sampler)
            target_loader_test = return_test_data(args, test_sampler)
            data_iter_t_unl = iter(target_loader_unl_1)
            len_train_target_semi = len(target_loader_unl_1)
            data_iter_t_lab = iter(target_loader_lab)
            len_train_target_lab = len(target_loader_lab)		
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, ste,
                                       init_lr=args.lr, update_pseudo=args.update_pseudo)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, ste,
                                       init_lr=args.lr, update_pseudo=args.update_pseudo)
        lr = optimizer_f.param_groups[0]['lr']
    #   lr = 0.01
        if ste % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_s = next(data_iter_s)
        #im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        #gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_s,gt_labels_s = data_s[0].cuda(), data_s[1].cuda()

        if ste < args.update_pseudo:
            if ste % len_train_target_semi == 0:
                data_iter_t_unl = iter(target_loader_unl)
            data_t_unl = next(data_iter_t_unl)
            #im_data_tu.data.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
            im_data_tu = data_t_unl[0].cuda()

            zero_grad_all()
            data = im_data_s
            target = gt_labels_s
        else:
            if ste % len_train_target_semi == 0:
                data_iter_t_unl = iter(target_loader_unl_1)
            data_t_unl = next(data_iter_t_unl)
            im_data_tu = data_t_unl[0].cuda()
            if ste % len_train_target_lab == 0:
                data_iter_t_lab = iter(target_loader_lab)
            data_t_lab = next(data_iter_t_lab)
            im_data_t_lab, gt_labels_t_lab = data_t_lab[0].cuda(), data_t_lab[1].cuda()
            zero_grad_all()
            data = torch.cat((im_data_s, im_data_t_lab), 0)
            target = torch.cat((gt_labels_s, gt_labels_t_lab), 0)

        output = G(data)
        out1, _ = F1(output)
        #loss = criterion(out1, target)
        loss = CB_loss(target, out1, cls_num_list, len(class_list), loss_type="focal", beta=0.9999, gamma=2.0)
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        if not args.method == 'S+T':
            data_augmentation_reg(data_t_unl, F1, G, args)
            output = G(im_data_tu)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, ste, args)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             ste, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target,
                       ste, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if ste % args.log_interval == 0:
            print(log_train)
 
        if ste % args.save_interval ==0 and ste >0:
            G.train()
            F1.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, ste)))
                torch.save(F1.state_dict(),
                           os.path.join(checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, ste)))


        if ste % args.save_interval == 0 and ste > args.update_pseudo:
            loss_test, acc_test = test(target_loader_test)
            G.train()
            F1.train()
            if acc_test >= best_acc:
                best_acc = acc_test
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('current acc test %f' % best_acc)
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f \n' % (ste,best_acc_test))
        
 
def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t, gt_labels_t = data_t[0].cuda(), data_t[1].cuda()
            #im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            #gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1, norm_feat = F1(feat)
            output1 = norm_feat.mm(torch.transpose(F.normalize(F1.fc.weight, dim=1), 0, 1))
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size

def test1(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = 126
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    target_labels = []
    entr_ = []
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t, gt_labels_t = data_t[0].cuda(), data_t[1].cuda()
            #im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            #gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
          #  output1, _ = F1(feat)
            target_labels.extend(gt_labels_t.data.cpu().numpy())
            loss_t = entropy(F1, feat, args.lamda)
            entr_.append(loss_t.data.cpu().numpy())
        entr_ = np.asarray(entr_)
        samp_ = np.argsort(entr_)
        samp_ = samp_[126:]
        samp_ = samp_.astype(int)
        labels = np.asarray(target_labels)
        select_t_gt = labels[samp_]
        accuracy_pseudo1(labels, select_t_gt, samp_, args)
    #        output_all = np.r_[output_all, output1.data.cpu().numpy()]
         #   size += im_data_t.size(0)
         #   pred1 = output1.data.max(1)[1]
#            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
#                confusion_matrix[t.long(), p.long()] += 1
    
	
def get_proto(loader):

    G.eval()
    F1.eval()

    if "resnet" in args.net:
        protos = F1.fc2.weight.data.cpu().numpy()
    else:
        protos = F1.fc.weight.data.cpu().numpy()
    return protos
	

def eval_(loader, source_loader_feat):

    target_features = []
    target_labels = []
    query_features = []
    query_labels = []
    query_labels_pred = []
    G.eval()
    F1.eval()
    size = 0
    correct = 0

    if "resnet" in args.net:
        protos = F1.fc2.weight.data.cpu().numpy()
    else:
        protos = F1.fc.weight.data.cpu().numpy()
    np.save('./inter_features/prototypes', protos, allow_pickle=True, fix_imports=True)
    
    with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(loader)):
                im_data_t, gt_labels_t = data_t[0].cuda(), data_t[1].cuda()
                #im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
                #gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])

                feat = G(im_data_t)
                _, feats = F1(feat)
                target_features.append(feats)
                target_labels.extend(gt_labels_t.data.cpu().numpy())

            features = torch.cat(target_features).cpu()
            labels = np.asarray(target_labels)
            np.save('./inter_features/target_features', features, allow_pickle=True, fix_imports=True)
            np.save('./inter_features/target_labels', labels, allow_pickle=True, fix_imports=True)

            for batch_idx1, data_s in tqdm(enumerate(source_loader_feat)):
                im_data_s, gt_labels_s = data_s[0].cuda(), data_s[1].cuda()
                #im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
                #gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
                feat = G(im_data_s)
                output1, feats = F1(feat)
                query_features.append(feats)
                query_labels.extend(gt_labels_s.data.cpu().numpy())
                pred1 = output1.data.max(1)[1]
                query_labels_pred.extend(pred1.data.cpu().numpy())
            query_features = torch.cat(query_features).cpu()
            query_labels = np.asarray(query_labels)
            query_labels_pred = np.asarray(query_labels_pred)
            np.save('./inter_features/all_source_features', query_features, allow_pickle=True, fix_imports=True)
            np.save('./inter_features/all_source_gt', query_labels, allow_pickle=True, fix_imports=True)
            np.save('./inter_features/all_source_pred', query_labels_pred, allow_pickle=True, fix_imports=True)
                

train()
