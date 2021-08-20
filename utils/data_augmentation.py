import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from utils.augmentations.randaugment import RandAugmentMC, RandAugmentPC
from utils.augmentations.ctaugment import CTAugment
import numpy as np
from PIL import Image
# Might want to add resize image as done in other transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))



def TransformStrong(indexes, args, aug_policy="randaugment", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    root = './data/%s/' % args.dataset
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
        
    # Might want to add resize image as done in other transforms
    data_transforms = {
            'weak' : transforms.Compose([
	    ResizeImage(256),
	    transforms.RandomHorizontalFlip(),
	    transforms.RandomCrop(size=crop_size, padding=int(crop_size*0.125),padding_mode='reflect')]),

            'strong' : transforms.Compose([
	    ResizeImage(256),
	    transforms.RandomHorizontalFlip(),
	    transforms.RandomCrop(size=crop_size, padding=int(crop_size*0.125),padding_mode='reflect'), RandAugmentPC(n=2, m=10)]),

            'standard' : transforms.Compose([
	    ResizeImage(256),
	    transforms.CenterCrop(crop_size)]),

            'normalize' : transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=mean, std=std)])
        }
    images=[]
    for index in indexes:
        path = os.path.join(root, index)
        img = pil_loader(path) 
        strong = data_transforms['normalize'](data_transforms['strong'](img))
        images.append(strong)
    
    return torch.stack(images)

def data_augmentation_reg(data_t_unl, F1, G, args, thresh=0.95):
    im_data_tu_weak_aug = data_t_unl[0].cuda() 
    im_data_tu_strong_aug = TransformStrong(data_t_unl[2], args).cuda()
    pred_strong_aug, _ = F1(G(im_data_tu_strong_aug))
    with torch.no_grad():
        pred_weak_aug, _ = F1(G(im_data_tu_weak_aug))
    prob_weak_aug = F.softmax(pred_weak_aug,dim=1)
    mask_loss = prob_weak_aug.max(1)[0]>thresh
    pseudo_labels = pred_weak_aug.max(axis=1)[1]
    loss_pseudo_unl = torch.mean(mask_loss.int() * F.cross_entropy(pred_strong_aug, pseudo_labels))
    loss_pseudo_unl.backward(retain_graph=True)
    return
