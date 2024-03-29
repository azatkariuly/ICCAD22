'''
    created by: @azatkariuly
'''

#import mm_cuda_goliath1 as satmm_cuda
import satmm_cuda
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['birealnet18', 'birealnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

        #psum_q, s = quant_PTQ_cust(psum, nbits_psum)
        # psum_q, _ = quantizeLSQ_psum(psum, step_size_psum, nbits_psum)
        psum_q, _ = quantizeLSQ_psum(psum, 2**shift_value, nbits_psum)

        if SA:
            out = reduce(lambda x,y: (x+y).clip(min, max), psum_q.transpose(0,3)).squeeze().transpose(0,-1)
        else:
            out = OA(torch.sum(psum_q, axis=3).squeeze().transpose(1,-1), b=b)
        #out = cyclic_activation(out, k=2, b=b)
        return out*(2**shift_value) #step_size_psum
    #out = reduce(lambda x,y: (x+y).clip(min, max), psum.transpose(0,3)).squeeze().transpose(0,-1)
    #out = OA(torch.sum(psum, axis=3).squeeze().transpose(1,-1), b=b)
    #return out

def satconv2D(image, kernel, padding=0, stride=1, T=64, SA=False, b=8, signed=True,
              nbits_psum=8, step_size_psum=None):
    #B,Cin,H,W
    #Cout, Cin, H,W
    #B,Cout,H,W
    # Gather Shapes of Kernel + Image + Padding
    B,Cin,H,W=image.shape
    Cout,_,CH,CW = kernel.shape
    OH = (H - CH + 2 * padding) // stride + 1
    OW = (W - CW + 2 * padding) // stride + 1
    inp_unf = torch.nn.functional.unfold(image, (CH, CW),padding=padding,stride=stride)
    return satmm_cuda_temp(inp_unf.transpose(1, 2),kernel.view(Cout, -1).t(),
                           T=T, SA=SA, b=b, signed=signed, nbits_psum=nbits_psum,
                           step_size_psum=step_size_psum).reshape(B,Cout,OH,OW)

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

    vbar = round_pass((v/s).clamp(Qn, Qp))
    #vhat = vbar * s

    return vbar, s

def quant_PTQ_cust(v, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    delta = (v.max() - v.min())/(2**p - 1)
    v_q = (v/delta).round().clip(-2**(p-1), 2**(p-1)-1)

    return v_q, delta

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, **kwargs):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

        self.nbits_acc = kwargs['nbits_acc']
        self.nbits_psum = kwargs['nbits_acc'] - 3

        self.SA = kwargs['SA']
        #self.k = kwargs['k']

        #psum step sizes
        self.step_size_psum = kwargs['s'] #Parameter(torch.ones(1))

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        #y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        y = satconv2D(x, binary_weights, self.padding, self.stride, T=64, SA=self.SA, b=self.nbits_acc,
                      signed=True, nbits_psum=self.nbits_psum, step_size_psum=self.step_size_psum)

        return y*scaling_factor.reshape(1, -1, 1, 1)


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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_acc=8, s=8, SA=False, k=2):
        super(BasicBlock, self).__init__()

        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride, nbits_acc=nbits_acc, s=s, SA=SA, k=k)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.binary_activation(x)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, nbits_acc=8, s=8, SA=False, k=2):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], nbits_acc=nbits_acc, s=s, SA=SA, k=k)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, nbits_acc=nbits_acc, s=s, SA=SA, k=k)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, nbits_acc=nbits_acc, s=s, SA=SA, k=k)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, nbits_acc=nbits_acc, s=s, SA=SA, k=k)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, nbits_acc=8, s=8, SA=False, k=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, nbits_acc=nbits_acc, s=s, SA=SA, k=k))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_acc=nbits_acc, s=s, SA=SA, k=k))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model
