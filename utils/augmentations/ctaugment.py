# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from collections import namedtuple

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05)   ,
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


class CTAugment(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def rate_to_p(self, rate):
        p = rate + (1 - self.decay)  # Avoid to have all zero.
        p = p / p.max()
        p[p < self.th] = 0
        return p

    def policy(self, probe):
        kl = list(self.OPS.keys())
        v = []
        if probe:
            for _ in range(self.depth):
                k = random.choice(kl)
                bins = self.rates[k]
                rnd = np.random.uniform(0, 1, len(bins))
                v.append(self.OP(k, rnd.tolist()))
            return v
        for _ in range(self.depth):
            vt = []
            k = random.choice(kl)
            bins = self.rates[k]
            rnd = np.random.uniform(0, 1, len(bins))
            for r, bin in zip(rnd, bins):
                p = self.rate_to_p(bin)
                value = np.random.choice(p.shape[0], p=p / p.sum())
                vt.append((value + r) / p.shape[0])
            v.append(self.OP(k, vt))
        return v

    def update_rates(self, policy, proximity):
        for k, bins in policy:
            for p, rate in zip(bins, self.rates[k]):
                p = int(p * len(rate) * 0.999)
                rate[p] = rate[p] * self.decay + proximity * (1 - self.decay)


    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img



"""

OPS = {}
OP = namedtuple('OP', ('f', 'bins'))
Sample = namedtuple('Sample', ('train', 'probe'))

def register(*bins):
    def wrap(f):
        OPS[f.__name__] = OP(f, bins)
        return f

    return wrap


def apply(x, ops):
    if ops is None:
        return x
    y = Image.fromarray(np.round(127.5 * (1 + x)).clip(0, 255).astype('uint8'))
    for op, args in ops:
        y = OPS[op].f(y, *args)
    return np.asarray(y).astype('f') / 127.5 - 1


class CTAugment(object):
    def __init__(self, depth=2, th=0.85, decay=0.99):
        self.decay = decay
        self.depth = depth
        self.th = th
        self.rates = {}
        self.OPS = {}
        for idx, x in enumerate(fixmatch_augment_pool()):
            self.OPS[x[0].__name__] = OP

        for k, op in self.OPS.items():
            self.rates[k] = tuple([np.ones(x, 'f') for x in op.bins])

    def rate_to_p(self, rate):
        p = rate + (1 - self.decay)  # Avoid to have all zero.
        p = p / p.max()
        p[p < self.th] = 0
        return p

    def policy(self, probe):
        kl = list(self.OPS.keys())
        v = []
        if probe:
            for _ in range(self.depth):
                k = random.choice(kl)
                bins = self.rates[k]
                rnd = np.random.uniform(0, 1, len(bins))
                v.append(self.OP(k, rnd.tolist()))
            return v
        for _ in range(self.depth):
            vt = []
            k = random.choice(kl)
            bins = self.rates[k]
            rnd = np.random.uniform(0, 1, len(bins))
            for r, bin in zip(rnd, bins):
                p = self.rate_to_p(bin)
                value = np.random.choice(p.shape[0], p=p / p.sum())
                vt.append((value + r) / p.shape[0])
            v.append(self.OP(k, vt))
        return v

    def update_rates(self, policy, proximity):
        for k, bins in policy:
            for p, rate in zip(bins, self.rates[k]):
                p = int(p * len(rate) * 0.999)
                rate[p] = rate[p] * self.decay + proximity * (1 - self.decay)

    def stats(self):
        return '\n'.join('%-16s    %s' % (k, ' / '.join(' '.join('%.2f' % x for x in self.rate_to_p(rate)) for rate in self.rates[k])) for k in sorted(self.OPS.keys()))

"""
