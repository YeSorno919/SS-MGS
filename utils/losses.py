import torch
import torch.nn.functional as F
import numpy as np
import math
from math import exp
import math
from torch import nn
from torch.autograd import Variable

def gradient_loss_2d(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])


    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

def app_gradient_loss_2d(mask, s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])


    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx


    d = torch.mean(mask * (dx + dy ))
    return d / 2.0

def ncc_loss_2d(I, J, win=None):
    ndims = 2

    if win is None:
        win = [9,9]

    sum_filt = torch.ones([1, 1,  win[0], win[1]]).to("cuda")

    pad_no = math.floor(win[0]/2)

    stride = (1,1)
    padding = (pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums_2d(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return 1 - torch.mean(cc)

def compute_local_sums_2d(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = int(win[0] * win[1])
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def dice_coef_2d(y_true, y_pred):
    smooth = 1.
    a = torch.sum(y_true * y_pred, (2, 3))
    b = torch.sum(y_true**2, (2, 3))
    c = torch.sum(y_pred**2, (2, 3))
    dice = (2 * a + smooth) / (b + c + smooth)
    return torch.mean(dice)

def dice_loss_2d(y_true, y_pred):
    d = dice_coef_2d(y_true, y_pred)
    return 1 - d

def att_dice(y_true, y_pred):
    dice = dice_coef_2d(y_true, y_pred).detach()
    loss = (1 - dice) ** 2 *(1 - dice)
    return loss

def masked_dice_loss(y_true, y_pred, mask):
    smooth = 1.
    a = torch.sum(y_true * y_pred * mask, (2, 3, 4))
    b = torch.sum((y_true + y_pred) * mask, (2, 3, 4))
    dice = (2 * a) / (b + smooth)
    return 1 - torch.mean(dice)

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef_2d(y_true, y_pred)

def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth))

def mask_crossentropy(y_pred, y_true, mask):
    smooth = 1e-6
    return -torch.mean(mask * y_true * torch.log(y_pred+smooth))

def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return 1-_ssim(img1, img2, window, window_size, channel, size_average)



def cross_entropy_loss(input, target, weight=None, reduction='mean'): 
    return F.cross_entropy(input, target, weight=weight, reduction=reduction)

                                              

class DiceCELoss(nn.Module): 
    def __init__(self, lambda_ce=0.5, weight=None, reduction='mean'):
        super(DiceCELoss, self).__init__()
        self.lambda_ce = lambda_ce
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):

        assert input.shape[2:] == target.shape[2:], "Input and target spatial dimensions must match."

        dice = dice_loss_2d(target, F.softmax(input, dim=1))

        target_index = torch.argmax(target, dim=1)
        ce = cross_entropy_loss(input, target_index, weight=self.weight, reduction=self.reduction)
        
        return dice + self.lambda_ce * ce
    
