'''
    created by: @azatkariuly
'''

#import mm_cuda_goliath1 as satmm_cuda
import satmm_cuda
import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import reduce

import numpy as np

class satmm_psum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X, t):
        ctx.t = t
        out = satmm_cuda.forward_psum(A, X, t)
        ctx.save_for_backward(A, X)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.sum(axis=-1) / grad_output.shape[-1]
        A, X = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, X.T)
        grad_weight = torch.matmul(A.transpose(1,2), grad_output)
        return grad_input, grad_weight, None

def satmm_cuda_temp(A, X, T=64, SA=False, b=8, signed=True, nbits_psum=8, step_size_psum=None):
    width=2**b # 256
    max = (width >> signed) - 1 #127 or 255
    min = max - width + 1

    satmm_cuda_psum = satmm_psum.apply
    psum = satmm_cuda_psum(A.contiguous(),X.contiguous(), T)

    if step_size_psum is not None:
        N = psum.shape[3]

        shift_value = 4
        if b == 7:
            if N >= 35:
                shift_value = 3
            else:
                shift_value = 2
        if b == 6:
            if N >= 34:
                shift_value = 4
            else:
                shift_value = 3
        if b == 5:
            if N >= 10:
                shift_value = 4
            else:
                shift_value = 3
        if b == 4:
            if N >= 30:
                shift_value = 5
            else:
                shift_value = 4
        if b == 3:
            if N >= 27:
                shift_value = 6
            else:
                shift_value = 5
        if b == 2:
            if N >= 28:
                shift_value = 6
            else:
                shift_value = 5

        # psum_q, s = quant_PTQ_cust(psum, nbits_psum)
        # psum_q, s = quant_PTQ(psum, step_size_psum, nbits_psum)
        # psum_q, _ = quantizeLSQ_psum(psum, step_size_psum, nbits_psum)
        psum_q, _ = quantizeLSQ_psum(psum, 2**shift_value, nbits_psum)

        if SA:
            out = reduce(lambda x,y: (x+y).clip(min, max), psum_q.transpose(0,3)).squeeze().transpose(0,-1)
        else:
            out = OA(torch.sum(psum_q, axis=3).squeeze().transpose(1,-1), b=b)
        #out = cyclic_activation(out, k=2, b=b)
        return out*(2**shift_value) #s #(2**shift_value) #step_size_psum

    #out = reduce(lambda x,y: (x+y).clip(min, max), psum.transpose(0,3)).squeeze().transpose(0,-1)
    # out = OA(torch.sum(psum, axis=3).squeeze().transpose(1,-1), b=b)
    # return out

def satconv2D(image, kernel, padding=0, stride=1, T=64, SA=False, b=8, signed=True,
              nbits_psum=8, step_size_psum=None):
    #B,Cin,H,W
    #Cout, Cin, H,W
    #B,Cout,H,W
    # Gather Shapes of Kernel + Image + Padding
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding[0]) // stride[0] + 1
    OW = (W - CW + 2 * padding[1]) // stride[0] + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    return satmm_cuda_temp(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(),
                           T=T, SA=SA, b=b, signed=signed, nbits_psum=nbits_psum,
                           step_size_psum=step_size_psum).reshape(B,Cout,OH,OW)

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ_psum(v, s, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    #gradScaleFactor = 1.0 / math.sqrt(v.numel()*Qp)
    #s = grad_scale(s, gradScaleFactor)

    vbar = round_pass((v/s)) #.clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s

def quant_PTQ_cust(v, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    #delta = (v.max() - v.min())/(2**p - 1)
    delta = 128/(2**p - 1)
    v_q = (v/delta).round().clip(-2**(p-1), 2**(p-1)-1)

    return v_q, delta

def quant_PTQ(v, s, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)
    s = grad_scale(s, gradScaleFactor)

    v_q = round_pass((v/s).clamp(Qn, Qp))

    return v_q, s


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(BinarizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.nbits_acc = kwargs['nbits_acc']
        self.nbits_psum = kwargs['nbits_acc']

        self.T = kwargs['T']
        self.SA = kwargs['SA']
        #self.k = kwargs['k']

        #psum step sizes
        self.step_size_psum = kwargs['s'] #Parameter(torch.ones(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        '''
        if self.init_state == 0:
            self.step_size_psum.data.copy_(20 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits_psum - 1) - 1))
            self.init_state.fill_(1)
        '''
        # out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        out = satconv2D(input, self.weight, self.padding, self.stride,
                        T=self.T, SA=self.SA, b=self.nbits_acc, signed=True,
                        nbits_psum=self.nbits_psum, step_size_psum=self.step_size_psum)

        # out = OA(out.int(), b=self.nbits_acc).float() + out - out.int()

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        #WrapNet cyclic activation
        #r = regularizer(out, b=self.nbits_acc)
        #out = cyclic_activation(out, k=self.k, b=self.nbits_acc)

        return out

def OA(x, b=4):
    mask = (1 << b) - 1
    mask2 = 2**(b-1)

    Qn = -2**(b-1)
    Qp = 2**(b-1)-1

    upper = (x > Qp).float()
    lower = (x < Qn).float()
    middle = 1.0 - upper - lower

    out = x*middle

    out2 = (x*(upper+lower)).int()&mask

    upper2 = (out2 > Qp).float()
    lower2 = (out2 < Qn).float()
    middle2 = 1.0 - upper2 - lower2

    out3 = out2*middle2 + (out2-2*mask2)*upper2 + (out2+2*mask2)*lower2

    return out+out3

def OA_wrapnet(x, b=4):
    return (x+2**(b-1)).remainder(2**b) - 2**(b-1)

def cyclic_activation(m, k, b):
    #m = (z+2**(b-1)).remainder(2**b) - 2**(b-1) #OA

    Q = k*(2**(b-1))/(k+1)

    upper = (m > Q).float()
    lower = (m < -Q).float()
    middle = 1.0 - upper - lower

    return (k*(2**(b-1))-k*m)*upper + (-k*(2**(b-1))-k*m)*lower + m*middle

def regularizer(out, b):
    return torch.max(out.abs()-2**(b-1), torch.Tensor([0]).cuda()).sum()
