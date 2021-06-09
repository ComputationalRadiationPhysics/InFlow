
# coding: utf-8


######################################################################################
# 
##################Out of distribution detection using InFlow##########################
# This notebook will demonstrate the out-of-distribution (OoD) detection using InFlow 
#
######################################################################################

# import packges
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from tqdm import tqdm
import sys, os
from INN.modules import PermuteRandom, Concat
from INN.framework import InputNode, OutputNode, ConditionNode, Node, ReversibleGraphNet, GraphINN
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift
from torch import distributions
import torchvision
import torchvision.transforms as transforms
import data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from typing import Callable, Union, Tuple, Iterable, List
from torch import Tensor
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import tensorflow as tf
from sklearn import metrics
from ood_metrics import fpr_at_95_tpr
import math

#############################################################################################################################


# set random seed and hyperparameters
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(device)
sub_n = 10
p_value = 0.05
batch_size = 250
num_workers = 8
n_dim = 3   # number of dimensions of the RGB image
c_in = 3    # number of input channels
c_out = 3   # number of output channels
encoding_dim = 32
gpu_ids = [0]

#############################################################################################################################


# import CIFAR 10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = y_train.astype('int64').reshape(-1,)
y_test = y_test.astype('int64').reshape(-1,)

#############################################################################################################################


# choose the subset of CIFAR 10 training images as in-distribution samples for training
n_data = X_train.shape[0]
idx = np.random.choice(n_data, size=n_data // sub_n, replace=False)
idx_h0 = np.delete(np.arange(n_data), idx, axis=0)
X_ref = X_train[idx]
print(X_ref.shape)



#permute the CIFAR 10 channels to fit as a pytorch tensor
def permute_c(x):
    return np.transpose(x.astype(np.float32), (0, 3, 1, 2))

X_ref_pt = permute_c(X_ref)



#############################################################################################################################

# define encoder architecture
encoder_net = nn.Sequential(
    nn.Conv2d(3, 64, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(64, 128, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(128, 512, 4, stride=2, padding=0),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2048, encoding_dim)
).to(device).eval()

# define preprocessing function
preprocess_fn = partial(preprocess_drift, model=encoder_net, device=device, batch_size=512)

# initialise the attention mechanism 
cd = MMDDrift(X_ref_pt, backend='pytorch', p_val=.05, 
              preprocess_fn=preprocess_fn, n_permutations=100)


#############################################################################################################################
# import CIFAR training dataset
train_cifar10_dataset = torchvision.datasets.ImageFolder(root='/home/......../CIFAR 10/test/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

train_cifar10_loader = torch.utils.data.DataLoader(train_cifar10_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)


#############################################################################################################################
# import CIFAR testing dataset
test_cifar10_dataset = torchvision.datasets.ImageFolder(root='/home/......../CIFAR 10/test/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

test_cifar10_loader = torch.utils.data.DataLoader(test_cifar10_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)


#############################################################################################################################
# import CelebA testing dataset
test_celeba_dataset = torchvision.datasets.ImageFolder(root='/home/......../CelebA/test/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

test_celeba_loader = torch.utils.data.DataLoader(test_celeba_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)


#############################################################################################################################
# import SVHN testing dataset
test_svhn_dataset = torchvision.datasets.ImageFolder(root='/home/......../SVHN/test/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

test_svhn_loader = torch.utils.data.DataLoader(test_svhn_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)

#############################################################################################################################
# import MNIST testing dataset
test_mnist_dataset = torchvision.datasets.ImageFolder(root='/home/......../MNIST/test/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

test_mnist_loader = torch.utils.data.DataLoader(test_mnist_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)


#############################################################################################################################
# import FashionMNIST testing dataset
test_fashionmnist_dataset = torchvision.datasets.ImageFolder(root='/home/......../FashionMNIST/test/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

test_fashionmnist_loader = torch.utils.data.DataLoader(test_fashionmnist_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)



#############################################################################################################################
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#############################################################################################################################
#define the sub networks  's' and 't' architecture
def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 3, padding=1))

#############################################################################################################################
#construct the Invertible module for building the flow architecture
class InvertibleModule(nn.Module):
    def __init__(self, dims_in: Iterable[Tuple[int]],
                 dims_c: Iterable[Tuple[int]] = None):
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = list(dims_in)
        self.dims_c = list(dims_c)

    def forward(self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[Tensor], Tensor]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide forward(...) method")

    def jacobian(self, *args, **kwargs):
        raise DeprecationWarning("module.jacobian(...) is deprecated. "
                                 "module.forward(..., jac=True) returns a "
                                 "tuple (out, jacobian) now.")

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide output_dims(...)")

#############################################################################################################################
#construct the base coupling block
class _BaseCouplingBlock(InvertibleModule):
    def __init__(self, dims_in, dims_c=[],
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]  # if input is 3 channels then it would be 3

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        self.split_len1 = self.channels // 2                  # if input 3 channels then len1 = 1
        self.split_len2 = self.channels - self.channels // 2  # if input 3 channels then len2 = 2

        self.clamp = clamp

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]),             "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = (lambda u: 2. * (torch.sigmoid(u) - 0.5))
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        
        x1, x2= torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        s = c[0][0,0,0,0]
        
        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1, j1 = self._coupling1(s,x1, x2_c)

            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2, j2 = self._coupling2(s,x2, y1_c)
        else:
            # names of x and y are swapped for the reverse computation
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2, j2 = self._coupling2(s,x2, x1_c, rev=True)

            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1, j1 = self._coupling1(s,x1, y2_c, rev=True)

        return (torch.cat((y1, y2), 1),), j1 + j2

    def _coupling1(self,s, x1, u2, rev=False):
        raise NotImplementedError()

    def _coupling2(self, s,x2, u1, rev=False):
        raise NotImplementedError()

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims

#############################################################################################################################

#construct the InFlow coupling block
class InFlowCouplingBlock(_BaseCouplingBlock):
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet_s1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2)
        self.subnet_t1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2)
        self.subnet_s2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1)
        self.subnet_t2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1)

    def _coupling1(self, s, x1, u2, rev=False):      
        s2, t2 = self.subnet_s2(u2), self.subnet_t2(u2)
        s2 = self.clamp * self.f_clamp(s2)
        j1 = torch.sum(s2, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1
        else:
            y1 = torch.exp(s * s2) * x1 + s * t2
            return y1, j1

    def _coupling2(self, s, x2, u1, rev=False):
        s1, t1 = self.subnet_s1(u1), self.subnet_t1(u1)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = torch.sum(s1, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2
        else:
            y2 = torch.exp(s * s1) * x2 + s * t1
            return y2, j2

#############################################################################################################################
#Stack all the coupling blocks including the permute blocks and the conditional nodes
in1 = InputNode(3,32,32, name='input1')
cond = ConditionNode(3,32,32, name='Condition')
layer1 = Node(in1,InFlowCouplingBlock,{'subnet_constructor':subnet_conv, 'clamp':2.0},conditions=cond,name=F'coupling_{0}')
layer2 = Node(layer1, PermuteRandom,{'seed':0}, name=F'permute_{0}')
layer3 = Node(layer2,InFlowCouplingBlock,{'subnet_constructor':subnet_conv, 'clamp':2.0},conditions=cond,name=F'coupling_{1}')
layer4 = Node(layer3,PermuteRandom,{'seed':1},name=F'permute_{1}')
out1 = OutputNode(layer4, name='output1')

model = GraphINN([in1, cond, layer1, layer2, layer3, layer4, out1]).cuda()

#############################################################################################################################

#Load thetrained InFlow Model 
state_dicts = torch.load('/home/......./ckptdir/199.pt')
model.load_state_dict(state_dicts['net'])

#############################################################################################################################

# Define the inference scheme for unknown test samples
dist = math.sqrt(2) * tf.math.erfinv(1- p_value).numpy()
D = 3 * 32 * 32
prior = distributions.MultivariateNormal(torch.zeros(D).cuda(),
                                         dist * torch.eye(D).cuda())
def get_loss_vals_trained(loader, net, cd, batch_size):
    loss_vals = []
    cx = []
    with torch.no_grad():
        with tqdm(total=len(loader.dataset)) as progress_bar:
            for x, _ in loader:
                
                x_numpy = x.detach().cpu().numpy()
                pval = cd.predict(x_numpy, return_p_val=True)['data']['p_val']
                #print(pval)
                if pval < 0.05:
                    x_c = torch.zeros(batch_size,3,32,32).cuda() 
                    cx.append(np.zeros(batch_size))
                    x = x.cuda()
                    # Forward step:
                    z, log_jac_det = model(x,x_c)
                    z = z.reshape((z.shape[0], -1))
                    prior_ll = prior.log_prob(z)
                    losses = prior_ll
                    loss_vals.extend([loss.item() for loss in losses])
                    progress_bar.update(x.size(0))                    
                else:
                    x_c = torch.ones(batch_size,3,32,32).cuda() 
                    cx.append(np.ones(batch_size))
                    x = x.cuda()
                    # Forward step:
                    z, log_jac_det = model(x,x_c)
                    z = z.reshape((z.shape[0], -1))
                    prior_ll = prior.log_prob(z)
                    losses = prior_ll - log_jac_det
                    loss_vals.extend([loss.item() for loss in losses])
                    progress_bar.update(x.size(0))          
                    
    return np.array(loss_vals), cx

#############################################################################################################################
#define the inference scheme for the in-distribution CIFAR 10 samples
def get_loss_vals_ones(loader, net, cd, batch_size):
    loss_vals = []
    cx = []
    with torch.no_grad():
        with tqdm(total=len(loader.dataset)) as progress_bar:
            for x, _ in loader:
                x_numpy = x.detach().cpu().numpy()
                pval = cd.predict(x_numpy, return_p_val=True)['data']['p_val']
                x_c = torch.ones(batch_size,3,32,32).cuda() 
                cx.append(np.ones(batch_size))
                x = x.cuda()
                # Forward step:
                z, log_jac_det = model(x,x_c)
                z = z.reshape((z.shape[0], -1))
                prior_ll = prior.log_prob(z)
                losses = prior_ll - log_jac_det
                loss_vals.extend([loss.item() for loss in losses])
                progress_bar.update(x.size(0))          
                    
    return np.array(loss_vals), cx
#############################################################################################################################


#Start inference and calculate the log-likelihood scores
train_cifar10_loss_vals_attention, cx = get_loss_vals_ones(train_cifar10_loader, model, cd, batch_size)
test_cifar10_loss_vals_attention, cx = get_loss_vals_trained(test_cifar10_loader, model, cd, batch_size)
test_celeba_loss_vals_attention, cx = get_loss_vals_trained(test_celeba_loader, model, cd, batch_size)
test_svhn_loss_vals_attention, cx = get_loss_vals_trained(test_svhn_loader, model, cd, batch_size)
test_mnist_loss_vals_attention, cx = get_loss_vals_trained(test_mnist_loader, model, cd, batch_size)
test_fashionmnist_loss_vals_attention, cx = get_loss_vals_trained(test_fashionmnist_loader, model, cd, batch_size)
#############################################################################################################################

#definition for plotting the ROC Curve 
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
#############################################################################################################################


#Calculate the AUCROC, AUCPR and FPR95 scores for CelebA test dataset (Note: Follow similar steps for the other datasets 
combined = np.concatenate((train_cifar10_loss_vals_attention, test_celeba_loss_vals_attention))
label_1 = np.ones(len(train_cifar10_loss_vals_attention))
label_2 = np.zeros(len(test_celeba_loss_vals_attention))
label = np.concatenate((label_1, label_2))

fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
precision, recall, thresholds_ = metrics.precision_recall_curve(label, combined, pos_label=0)

plot_roc_curve(fpr, tpr)

rocauc = metrics.auc(fpr, tpr)
aucpr = metrics.auc(recall, precision)

print('AUCROC for CelebA OOD: ', rocauc)
print('AUCPR for CelebA OOD: ', aucpr)
print('FPR95 for CelebA OOD: ', 1- fpr_at_95_tpr(combined, label))


#############################################################################################################################
