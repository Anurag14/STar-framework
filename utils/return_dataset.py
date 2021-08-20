import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist, return_number_of_label_per_class
from pdb import set_trace as bkpt

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def return_pseudo(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
   
    if args.setting == 'SSDA':
 
        image_set_file_t_lab = \
            os.path.join(base_path,
                         'pseudo_images_' + 
                         args.source + '_' + args.target + '_%d.txt' % (args.num))
    else:
        image_set_file_t_lab = \
            os.path.join(base_path,
                         'pseudo_images_' +
                         args.source + '_' + args.target + '.txt')
        
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_lab = Imagelists_VISDA(image_set_file_t_lab, root=root,
                                      transform=data_transforms['val'])
    if args.net == 'alexnet':
        bs =32 
    else:
        bs = 24
    target_loader_lab = \
        torch.utils.data.DataLoader(target_dataset_lab,
                                    batch_size=min(bs, len(target_dataset_lab)),
                                    num_workers=0,
                                    shuffle=True, drop_last=True)
    return target_loader_lab
   
def return_pseudo1(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    if args.setting == 'SSDA':

        image_set_file_t_lab = \
            os.path.join(base_path,
                         'mean_select_images_' + args.source + '_' + args.target + '_%d.txt' % (args.num))
    else:
        image_set_file_t_lab = \
            os.path.join(base_path,
                        'mean_select_images_' + args.source + '_' + args.target + '.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_lab = Imagelists_VISDA(image_set_file_t_lab, root=root,
                                      transform=data_transforms['val'])
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_lab = \
        torch.utils.data.DataLoader(target_dataset_lab,
                                    batch_size=min(bs, len(target_dataset_lab)),
                                    num_workers=0,
                                    shuffle=True, drop_last=True)
    return target_loader_lab

def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))    

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, test=True,
                                          transform=data_transforms['val'])
    t_length = len(target_dataset_unl)
    class_list = return_classlist(image_set_file_s)
    class_num_list = return_number_of_label_per_class(image_set_file_s,len(class_list)) 
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader_unl, class_list, class_num_list, t_length

def return_labeled_data(args, sampler, tar_sampler):
    
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
	}
	
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, test=True,
                                      transform=data_transforms['val'])

    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader = \
    torch.utils.data.DataLoader(target_dataset_unl, sampler=tar_sampler,
                                batch_size=min(bs, len(target_dataset_unl)), num_workers=0,
                                drop_last=True) #shuffle=True
    target_loader_unl = \
    torch.utils.data.DataLoader(target_dataset_unl, sampler=sampler,
                                batch_size=bs * 2, num_workers=0,
                                drop_last=True) #shuffle=True
    
    return target_loader, target_loader_unl
		
def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))

    image_set_file_source = os.path.join(base_path,
                                         'labeled_source_images_' +
                                         args.source + '.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)

    num_images = len(target_dataset_unl)
    source_dataset = Imagelists_VISDA(image_set_file_source, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_test)

    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)

    source_loader = \
        torch.utils.data.DataLoader(source_dataset,
                                    batch_size=bs*2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, source_loader, class_list
	
def return_eval_data(args, eval_sampler):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_eval = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['val'],
                                          test=True)
    num_images = len(target_dataset_eval)

    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_eval = \
        torch.utils.data.DataLoader(target_dataset_eval, sampler= eval_sampler,
                                    batch_size=bs * 2, num_workers=3,
                                    drop_last=False)                           #shuffle=False

    return target_loader_eval

def return_test_data(args, sampler='None'):

    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, test=True,
                                           transform=data_transforms['test'])

    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
  
    if sampler == 'None':
        target_loader_test = \
            torch.utils.data.DataLoader(target_dataset_test,
                                        batch_size=bs * 2, num_workers=0,
                                        drop_last=False)
    else:
        target_loader_test = \
            torch.utils.data.DataLoader(target_dataset_test, sampler= sampler,
                                        batch_size=bs * 2, num_workers=0,
                                        drop_last=False)	    
    return target_loader_test
