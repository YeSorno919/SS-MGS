import math
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from scipy.ndimage import morphology
import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

def dice(pre, gt):
    tmp = pre + gt
    a = np.sum(np.where(tmp == 2, 1, 0))
    b = np.sum(pre)
    c = np.sum(gt)
    dice = (2*a)/(b+c+1e-6)
    return dice

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

def EMA(model_A, model_B, alpha=0.999):
    """
    Momentum update of the key encoder
    """
    for param_B, param_A in zip(model_B.parameters(), model_A.parameters()):
        param_A.data = alpha*param_B.data + (1-alpha)*param_B.data
    return model_A

def adjust_learning_rate(optimizer, epoch, epochs, lr, schedule, is_cos=False):
    if is_cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def delete_except_last_n(directory, n=10): 

    dir_path = Path(directory)

    if not dir_path.exists(): 
        print(f"Directory {directory} does not exist.")
        return 

    entries = list(dir_path.iterdir()) 

    entries.sort(key=os.path.getmtime, reverse=True) 

    for entry in entries[n:]: 
        try: 

            if entry.is_file() or entry.is_symlink():
                os.remove(entry)

            elif entry.is_dir(): 
                shutil.rmtree(entry) 
        except Exception as e: 
            print(f'Failed to delete {entry}. Reason: {e}') 
            
def draw_gland_over_image_and_fix_moving_fixed_imgs_without_mask(save_path,moving_gland_image,moving_mask_image,fixed_gland_image,fixed_mask_image,moving_name,fixed_name,overlay_color,alpha):
        overlay_color1  = overlay_color[0]
        overlay_color2 = overlay_color[1]
        moving_mask_image = np.where(moving_mask_image <= 0, 0, moving_mask_image)
        moving_mask_image = np.where(moving_mask_image> 0, 255, moving_mask_image)

        moving_gland_image_color = cv2.cvtColor(moving_gland_image, cv2.COLOR_GRAY2BGR)

        moving_overlay_image = np.full_like(moving_gland_image_color, overlay_color1, dtype=np.uint8)

        moving_overlay_channels = cv2.split(moving_overlay_image)

        moving_overlayed_channels = []

        for channel in moving_overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, moving_mask_image)
            moving_overlayed_channels.append(overlayed_channel)

        moving_overlayed_image = cv2.merge(moving_overlayed_channels)

        moving_result_image = cv2.addWeighted(moving_gland_image_color, 1, moving_overlayed_image, alpha, 0)
    
        '''
        fix img
        '''
        fixed_mask_image = np.where(fixed_mask_image <= 0, 0, fixed_mask_image)
        fixed_mask_image = np.where(fixed_mask_image> 0, 255, fixed_mask_image)

        fixed_gland_image_color = cv2.cvtColor(fixed_gland_image, cv2.COLOR_GRAY2BGR)

        fixed_overlay_image = np.full_like(fixed_gland_image_color, overlay_color2, dtype=np.uint8)

        fixed_overlay_channels = cv2.split(fixed_overlay_image)

        fixed_overlayed_channels = []

        for channel in fixed_overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, fixed_mask_image)
            fixed_overlayed_channels.append(overlayed_channel)

        fixed_overlayed_image = cv2.merge(fixed_overlayed_channels)

        fixed_result_image = cv2.addWeighted(fixed_gland_image_color, 1, fixed_overlayed_image, alpha, 0)

        result = cv2.addWeighted(moving_result_image, 0.5, fixed_result_image, 0.5, 0)

        result_image_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland')):
            os.makedirs(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland'))
        cv2.imwrite(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland', moving_name + '_' + fixed_name + '.jpg'), result_image_bgr)
        return  result_image_bgr