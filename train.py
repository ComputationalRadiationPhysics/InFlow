

# coding: utf-8


######################################################################################
# 
##################Out of distribution detection using InFlow##########################
# This notebook will demonstrate the training mechanism for InFlow 
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
from torch import distributions
import torchvision
import torchvision.transforms as transforms
import data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from typing import Callable, Union, Tuple, Iterable, List
from torch import Tensor
import tensorflow as tf
import math
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time


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
# import CIFAR training dataset

train_cifar10_dataset = torchvision.datasets.ImageFolder(root='/home/...../CIFAR 10/train/',
                                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.CenterCrop(32),
                                                                                      transforms.ToTensor(),]))

train_cifar10_loader = torch.utils.data.DataLoader(train_cifar10_dataset,batch_size=batch_size,shuffle=False,
                                                  num_workers=num_workers,pin_memory=True)


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

# Define the training scheme
def train(i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0
    
    t_start = time.time()
    
    for x,_ in train_cifar10_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break
        x_numpy = x.detach().cpu().numpy()
        x_numpy = np.transpose(x_numpy, (0, -1, 1, 2))
        x_numpy = np.transpose(x_numpy,(0, -1, 1, 2))
        x_c = torch.ones(batch_size,3,32,32)
        optimizer.zero_grad()

        # Forward step: # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_jac_det = model(x.cuda(),x_c.cuda())
        
        #calculate the negative log-likelihood of the model with a standard normal prior
        loss = 0.5*torch.sum(z**2, (1,2,3)) - log_jac_det

        loss = loss.mean() / n_dim
        l_tot += loss.data.item()
            
        loss.backward()
        optimizer.step()
        
    return l_tot / batch_idx


#############################################################################################################################
#start training
ML_loss = []
save_freq = 50      #frequency of saving ckpts
n_epochs = 200
n_its_per_epoch = 100
lr = 1e-4
l2_reg = 2e-5

trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.99), eps=1e-8, weight_decay=l2_reg)

ckptdir = '/home/...../ckptdir/'
model     
try:
    t_start = time.time()
    for i_epoch in range(n_epochs):
        ml_loss = train(i_epoch)
        ML_loss.append(ml_loss)
        
        if(i_epoch % 1 == 0):
             print("Epoch: [%2d], ml_loss: [%.8f]"
                      %(i_epoch, ml_loss))
            
        if (i_epoch %  save_freq == 0):
            print('Saving...')
            state = {
                'net': model.state_dict(),
                'epoch': i_epoch,
        }
        os.makedirs(ckptdir, exist_ok=True)
        torch.save(state, os.path.join(ckptdir, str(i_epoch) + '.pt'))

except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time.time()-t_start)/60:.2f} minutes\n")


##############################################################################################################################
# plot the likelihood loss curve
loss_curve = np.asarray(ML_loss)
plt.plot(loss_curve)
plt.xlabel('epoch')
plt.ylabel('ML loss')
plt.title('Maximum likelihood (Minimise Negative Log likelihood)')
plt.savefig('/home/...../Loss_Curve.png', dpi=200)

#############################################################################################################################