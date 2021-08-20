from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(self, x,lambd):
        result = x.view_as(x)
        self.lambd = lambd
        #self.save_for_backward(result)
        return result
    @staticmethod
    def backward(self, grad_output):
        lambd = self.lambd
        #print(self.saved_tensors)
        #input = self.saved_tensors
        return (grad_output * -lambd), None

"""
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    #@staticmethod
    def forward(self, x):
        return x.view_as(x)
    #@staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)
"""


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x,lambd)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x

class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out, x


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out, x

class Predictor_feat(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_feat, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out, x

class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 2)

    def forward(self, x, reverse=False, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        #x_out = F.sigmoid(self.fc3_1(x))
        #return x_out
        x_out = self.fc3_1(x)
        return x_out


class Discriminator_classwise(nn.Module):
    def __init__(self, inc=4096, num_class=126):
        super(Discriminator_classwise, self).__init__()
        self.classfier_list = []
        for _ in range(num_class):
            self.classfier_list.append(nn.Sequential(nn.Linear(inc,512),nn.Linear(512,512),nn.Linear(512,2)).cuda())

    def forward(self, x, reverse=False, eta=1.0, choose_class=0):
        if reverse:
            x = grad_reverse(x, eta)
        which_classifier = self.classfier_list[choose_class]
        x_out = which_classifier(x)
        return x_out


class Predictor_deep_new(nn.Module):
    def __init__(self, num_class=64, temp=0.05):
        super(Predictor_deep_new, self).__init__()
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out
