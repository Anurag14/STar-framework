# Defining the augmentation schemes going to be used
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from collections import namedtuple
import torch
from torchvision import transforms





def _enhance(x, op, level):
    return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
    return Image.blend(x, op(x), level)


def _filter(x, op, level):
    return Image.blend(x, x.filter(op), level)


def autocontrast(x, level):
    return _imageop(x, ImageOps.autocontrast, level)


def blur(x, level):
    return _filter(x, ImageFilter.BLUR, level)


def brightness(x, brightness):
    return _enhance(x, ImageEnhance.Brightness, brightness)


def color(x, color):
    return _enhance(x, ImageEnhance.Color, color)


def contrast(x, contrast):
    return _enhance(x, ImageEnhance.Contrast, contrast)


def cutout(x, level):
    """Apply cutout to pil_img at the specified level."""
    size = 1 + int(level * min(x.size) * 0.499)
    img_height, img_width = x.size
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
    pixels = x.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (127, 127, 127)  # set the color accordingly
    return x


def equalize(x, level):
    return _imageop(x, ImageOps.equalize, level)


def invert(x, level):
    return _imageop(x, ImageOps.invert, level)


def identity(x):
    return x


def posterize(x, level):
    level = 1 + int(level * 7.999)
    return ImageOps.posterize(x, level)


def rescale(x, scale, method):
    s = x.size
    scale *= 0.25
    crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
    methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
    method = methods[int(method * 5.99)]
    return x.crop(crop).resize(x.size, method)


def rotate(x, angle):
    angle = int(np.round((2 * angle - 1) * 45))
    return x.rotate(angle)


def sharpness(x, sharpness):
    return _enhance(x, ImageEnhance.Sharpness, sharpness)


def shear_x(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


def shear_y(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


def smooth(x, level):
    return _filter(x, ImageFilter.SMOOTH, level)


def solarize(x, th):
    th = int(th * 255.999)
    return ImageOps.solarize(x, th)


def translate_x(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


def translate_y(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


    
global_augs_dict_strong = {'translate_x': translate_x, 'translate_y': translate_y, 'solarize': solarize, 'smooth': smooth, 'shear_x': shear_x, 'shear_y': shear_y,
                    'sharpness': sharpness, 'rotate': rotate, 'autocontrast': autocontrast, 'blur': blur, 'brightness': brightness, 'color': color,
                    'contrast': contrast, 'equalize': equalize, 'invert': invert, 'identity': identity, 'posterize': posterize}
global_augs_dict_weak = {}

augs = list(global_augs_dict_strong.keys())


def process_batch(batch,label = True):
    if label:
        for image in batch:
            aug = random.choice(augs)
            # Need to convert tensor to PIL image and back
            image = transforms.ToPILImage()(image.cpu()).convert("RGB")
            image = global_augs_dict_strong[aug](image)
            image = transforms.ToTensor()(image).cuda()
            return image
    else:
        print("no label")
    return batch
