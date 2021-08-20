# Defining the augmentation schemes going to be used
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from collections import namedtuple
import torch
from torchvision import transforms

# Defining custom augmentation class, so that we can inherit this class for trying different augmentation schemes 
class Augmentation:
    def __init__(self, level=0.5, scale = 0.75, method = 0, shear = 0.2, th = 0.5, delta = 0.2, angle = 30):    
        self.level = level
        self.scale = scale
        self.method = method
        self.shear = shear
        self.th = th
        self.delta = delta
        self.angle = angle
        self.range1 = [0,1]
        self.range_cutout = [0,0.5]
        self.range_angle = [-45,45]
        self.range_posterize = [1,8]
        self.range_rescale = [0.5,1]
        self.range_shear_translate = [-0.3,0.3]
        # Defining dictionaries for strong and weak augmentations
        self.global_augs_dict_strong = {'translate_x': self.translate_x, 'translate_y': self.translate_y, 'solarize': self.solarize, 'smooth': self.smooth, 'shear_x': self.shear_x, 'shear_y': self.shear_y,
                            'sharpness': self.sharpness, 'rotate': self.rotate, 'autocontrast': self.autocontrast, 'blur': self.blur, 'brightness': self.brightness, 'color': self.color,
                            'contrast': self.contrast, 'equalize': self.equalize, 'invert': self.invert, 'identity': self.identity, 'posterize': self.posterize}

        self.global_augs_dict_weak = {'flip_ud': self.flip_ud, 'translate_x': self.translate_x, 'translate_y': self.translate_y}
    
        self.strong_augs = list(self.global_augs_dict_strong.keys())
        self.weak_augs = list(self.global_augs_dict_weak.keys())



    def _enhance(self, x, op, level):
        return op(x).enhance(0.1 + 1.9 * level)


    def _imageop(self,x,op, level):
        return Image.blend(x, op(x), level)


    def _filter(self, x, op, level):
        return Image.blend(x, x.filter(op), level)

    def flip_ud(self,x):
        return ImageOps.flip(x)

    def autocontrast(self,x):
        return self._imageop(x, ImageOps.autocontrast, self.level)


    def blur(self,x):
        return self._filter(x, ImageFilter.BLUR, self.level)


    def brightness(self, x):
        return self._enhance(x, ImageEnhance.Brightness, self.level)


    def color(self,x):
        return self._enhance(x, ImageEnhance.Color, self.level)


    def contrast(self,x):
        return self._enhance(x, ImageEnhance.Contrast, self.level)

    def equalize(self,x):
        return self._imageop(x, ImageOps.equalize, self.level)


    def invert(self,x):
        return self._imageop(x, ImageOps.invert, self.level)


    def identity(self, x):
        return x


    def posterize(self, x):
        level = 1 + int(self.level * 7.999)
        return ImageOps.posterize(x, level)


    def rescale(self,x):
        s = x.size
        scale = self.scale * 0.25
        crop = (scale, scale, s[0] - scale, s[1] - scale)
        methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
        method = methods[int(self.method * 5.99)]
        return x.crop(crop).resize(x.size, method)


    def rotate(self, x):
        angle = int(np.round((2 * self.angle - 1) * 45))
        return x.rotate(angle)


    def sharpness(self, x):
        return self._enhance(x, ImageEnhance.Sharpness, self.level)


    def shear_x(self,x):
        shear = (2 * self.shear - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


    def shear_y(self, x):
        shear = (2 * self.shear - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


    def smooth(self, x):
        return self._filter(x, ImageFilter.SMOOTH, self.level)


    def solarize(self, x):
        th = int(self.th * 255.999)
        return ImageOps.solarize(x, th)


    def translate_x(self, x):
        delta = (2 * self.delta - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))

    def translate_y(self, x):
        delta = (2 * self.delta - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


def process_batch( batch, augment, label = True):
    if label:
        batch_strong = torch.zeros(batch.size())
        for i, image in enumerate(batch):
            aug = random.choice(augment.strong_augs)
            # Need to convert tensor to PIL image and back
            image_strong = transforms.ToPILImage()(image.cpu()).convert("RGB")
            aug_fn = augment.global_augs_dict_strong[aug]
            image_strong = aug_fn(image_strong)
            image_strong = transforms.ToTensor()(image_strong).cuda()
            batch_strong[i] = image_strong
        return batch_strong
    else:
        batch_strong = torch.zeros(batch.size())
        batch_weak = torch.zeros(batch.size())
        for i, image in enumerate(batch):
            aug = random.choice(augment.strong_augs)
            # Need to convert tensor to PIL image and back
            image_strong = transforms.ToPILImage()(image.cpu()).convert("RGB")
            aug_fn = augment.global_augs_dict_strong[aug]
            image_strong = aug_fn(image_strong)
            image_strong = transforms.ToTensor()(image_strong).cuda()
            batch_strong[i] = image_strong
        
        for i, image in enumerate(batch):
            aug = random.choice(augment.weak_augs)
            # Need to convert tensor to PIL image and back
            image_weak = transforms.ToPILImage()(image.cpu()).convert("RGB")
            aug_fn = augment.global_augs_dict_weak[aug]
            image_weak = aug_fn(image_weak)
            image_weak = transforms.ToTensor()(image_weak).cuda()
            batch_weak[i] = image_weak
        
        return batch_strong, batch_weak
