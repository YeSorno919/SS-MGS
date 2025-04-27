from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from  PIL import Image
from models.Reg_network import UNet_reg_with_se_2pam as UNet_Reg
from utils.STN import SpatialTransformer_2d, Re_SpatialTransformer_2d
from utils.dataloader_5fold_registration import Dataset_train_reg as Dataset_train
from utils.dataloader_5fold_registration import Dataset_reg_test as Dataset_reg
from utils.losses import gradient_loss_2d, ncc_loss_2d,ssim
from utils.utils import AverageMeter,delete_except_last_n
import json
from tqdm import tqdm
import os
import shutil
import torch.nn.functional as F
import random

class Reg_MGD(object):
    def __init__(self,
                foldnum,
                n_channels=1,
                n_classes=2,
                lr=1e-4,
                epoches=200,
                iters=200,
                batch_size=1,
                test_batch_size=1,
                subject_num = -1,
                data_dir = 'your data file dir',
                with_fix = True, 
                datajsonpath = '',
                with_fix_json_path = '',
                checkpoint_dir='checkpoin_output_dir',
                result_dir='result_output_dir',
                is_save_flow = False,
                is_save_val_img = False,
                cal_ncc_rate = 1,
                cal_ssim_rate = 1,
                cal_smooth_rate = 0.1,
                pretrainRegmodel='',
                save_best_mode_number = 10, 
                device = 0
                 ):
        super(Reg_MGD, self).__init__()
        self.foldnum=foldnum
        self.save_best_mode_number = save_best_mode_number
        self.pretrainRegmodel=pretrainRegmodel
        
        # initialize parameters
        self.is_save_val_img = is_save_val_img
        self.is_save_flow =is_save_flow
        self.n_classes = n_classes
        self.epoches = epoches
        self.batch_size =batch_size
        self.iters = iters
        self.lr = lr
        self.data_dir = data_dir
        self.datajsonpath = datajsonpath
        self.device=device
        self.cal_ncc_rate = cal_ncc_rate
        self.cal_ssim_rate = cal_ssim_rate
        self.cal_smooth_rate=cal_smooth_rate
        self.results_dir = result_dir                                                                                         
    
        self.model_name = 'The experiment name of the registration model'
        self.model_name =self.model_name + '/{}/'.format(self.foldnum)
        self.checkpoint_dir = os.path.join(checkpoint_dir,self.model_name)


        self.test_batch_size = test_batch_size
        self.with_fix =with_fix
        self.with_fix_json_path = with_fix_json_path
        
        if self.with_fix:
            self.datajsonpath = self.with_fix_json_path
        self.subject_num = subject_num
        self.writer = SummaryWriter(log_dir= self.results_dir+'/'+self.model_name+'/logs')
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        current_script_path = os.path.abspath(__file__)
        shutil.copy2(current_script_path, self.checkpoint_dir)

        self.stn = SpatialTransformer_2d() # Spatial Transformer
        self.lstn = SpatialTransformer_2d()
        self.rstn = Re_SpatialTransformer_2d() # Spatial Transformer-inverse
        self.rlstn = Re_SpatialTransformer_2d()
        self.softmax = nn.Softmax(dim=1)
        
        self.Reger = UNet_Reg(n_channels=n_channels)


        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()


        # initialize optimizer
        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr,weight_decay=1e-6)

        with open(self.datajsonpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)


        train_dataset = Dataset_train(self.data_dir,data["train"], self.n_classes, subject_num = self.subject_num,with_fix=self.with_fix)
        self.dataloader_train = DataLoader(train_dataset, batch_size=self.batch_size)
        test_dataset_reg = Dataset_reg(self.data_dir,data["val"], self.n_classes,with_fix=self.with_fix)
        self.dataloader_val_reg = DataLoader(test_dataset_reg, batch_size=self.test_batch_size)


        self.L_sim = ssim #
        self.L_sim_2  = ncc_loss_2d
        self.L_smooth = gradient_loss_2d

        # define loss log
        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')
        self.L_sim_2_log = AverageMeter(name='L_sim_2')
        self.L_Reg_log = AverageMeter(name='L_Reg') 
    
    def sigmoid_rampup(self, current): 
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        rampup_length = self.rampup_length
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


    def train_reg_iterator(self,labed_img, labed_lab, unlabed_img1, unlabed_img2,\
                l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,\
                l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,\
                epoch_num,iter_num):
        
        # train Reger
        for p in self.Reger.parameters():  # reset requires_grad           
            p.requires_grad = True  # they are set to True below in Reger update
        self.Reger.train()

        # random construct training data pairs
        rand = np.random.randint(low=0, high=3)
        if rand == 0:
            img1 = labed_img
            lab1 = labed_lab
            img2 = unlabed_img1
            lab2 = None

            img1_affine_to_img2 = l_u1_img
            img2_affine_to_img1 = u1_l_img


        elif rand == 1:
            img1 = labed_img
            lab1 = labed_lab
            img2 = unlabed_img2
            lab2 = None
            img1_affine_to_img2 = l_u2_img
            img2_affine_to_img1 = u2_l_img

        else: 
            img1 = unlabed_img2
            lab1 = None
            img2 = unlabed_img1
            lab2 = None
            img1_affine_to_img2 = u2_u1_img
            img2_affine_to_img1 = u1_u2_img
            
        if torch.cuda.is_available():
            img1_affine_to_img2 = img1_affine_to_img2.cuda()
            img2_affine_to_img1 = img2_affine_to_img1.cuda()

        warped_image, _, _, _, spatial_flow = self.Reger(img1_affine_to_img2, img2, lab1, lab2)
        
        # inverse deformation
        i_w_2_to_1, _, _, _, i_flow = self.Reger(img2_affine_to_img1, img1, lab2, lab1)

        # calculate loss
        loss_smooth = self.cal_smooth_rate*(self.L_smooth(spatial_flow) + self.L_smooth(i_flow))   # smooth loss ,
        self.L_smooth_log.update(loss_smooth.data, labed_img.size(0))

        loss_sim = self.cal_ssim_rate*(self.L_sim(warped_image, img2) + self.L_sim(i_w_2_to_1, img1))    # similarity loss
        loss_sim_2 =self.cal_ncc_rate*(self.L_sim_2(warped_image, img2) + self.L_sim_2(i_w_2_to_1, img1) )  # similarity loss
        self.L_sim_log.update(loss_sim.data, labed_img.size(0))
        self.L_sim_2_log.update(loss_sim_2.data, labed_img.size(0))

        loss_Reg = loss_smooth + loss_sim  + loss_sim_2
        self.L_Reg_log.update(loss_Reg.data,labed_img.size(0))
        

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad() 

    def train_reg_epoch(self, epoch):
        
        total_iters = self.iters  

        with tqdm(total=total_iters, desc='epoch', ncols=80) as pbar:
            for i in range(self.iters):
                # labed_img, labed_lab, unlabed_img1, unlabed_img2 = next(self.dataloader_train.__iter__())
                labed_img, labed_lab, unlabed_img1, unlabed_img2,\
                l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,\
                l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field = next(self.dataloader_train.__iter__())
                if torch.cuda.is_available():
                    labed_img = labed_img.cuda()
                    labed_lab = labed_lab.cuda()
                    unlabed_img1 = unlabed_img1.cuda()
                    unlabed_img2 = unlabed_img2.cuda()

                self.train_reg_iterator(
                                    labed_img, labed_lab, unlabed_img1, unlabed_img2,\
                                    l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,\
                                    l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,\
                                    epoch+1,i)
                res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                                'Iter: [%d/%d]' % (i + 1, self.iters),
                                self.L_smooth_log.__str__(),
                                self.L_sim_log.__str__(),
                                self.L_sim_2_log.__str__(),
                                self.L_Reg_log.__str__(),
                                ])
                print(res)
                pbar.update(1)

    def test_iterator_reg(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            # Reg
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        return w_m_to_f, w_label_m_to_f, flow

    def test_reg(self,epoch):
        reg_iou_list = []
        reg_recall_list = []
        reg_dice_list = []

        self.Reger.eval()

        total2_samples = len(self.dataloader_val_reg.dataset)  
        with tqdm(total=total2_samples, desc='Testing Regression', ncols=80) as pbar:
            for i, (mi, ml, fi, fl, l1_l2_img,l2_l1_img,l1_l2_seg,l2_l1_seg,l1_l2_displace_field,l2_l1_displace_field,name1, name2) in enumerate(self.dataloader_val_reg):

                if name1 is not name2:
                    if torch.cuda.is_available():
                        fi = fi.cuda()
                        fl = fl.cuda()
                        l1_l2_img =l1_l2_img.cuda()
                        l1_l2_seg = l1_l2_seg.cuda()

                    w_m_to_f, w_label_m_to_f, flow = self.test_iterator_reg(l1_l2_img, fi, l1_l2_seg, fl)

                    flow = flow.data.cpu().numpy()[0]
                    w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                    w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)

                    flow = flow.astype(np.float32)
                    w_m_to_f = w_m_to_f.astype(np.float32)
                    w_label_m_to_f = w_label_m_to_f.astype(np.int8)

                    if(self.is_save_flow == True and not os.path.exists(join(self.results_dir, self.model_name, 'flow'))):
                        os.makedirs(join(self.results_dir, self.model_name, 'flow'))
                    if(self.is_save_val_img == True and not os.path.exists(join(self.results_dir, self.model_name, 'w_m_to_f'))):
                        os.makedirs(join(self.results_dir, self.model_name, 'w_m_to_f'))

                    w_m_to_f = np.squeeze(w_m_to_f)
                    w_m_to_f = (w_m_to_f * 255).astype(np.uint8)
                    w_m_to_f = Image.fromarray(w_m_to_f)

                    if(self.is_save_val_img == True):
                        w_m_to_f.save(
                            join(self.results_dir, self.model_name, 'w_m_to_f', name1[0]+'_'+name2[0]+'.jpg'))

                    if(self.n_classes ==2 ):
                        if not os.path.exists(join(self.results_dir, self.model_name, 'w_label_m_to_f')):
                            os.makedirs(join(self.results_dir, self.model_name, 'w_label_m_to_f'))
                        w_label_m_to_f = np.squeeze(w_label_m_to_f) 
                        w_label_m_to_f = (w_label_m_to_f * 255).astype(np.uint8)
                        if( self.is_save_val_img == True):
                            w_label_m_to_f_image = Image.fromarray(w_label_m_to_f)
                            w_label_m_to_f_image.save(
                                join(self.results_dir, self.model_name, 'w_label_m_to_f',  name1[0]+'_'+name2[0]+'.jpg'))

                        w_label_m_to_f = np.where(w_label_m_to_f<=0,0,w_label_m_to_f)
                        w_label_m_to_f = np.where(w_label_m_to_f>0,1,w_label_m_to_f)
                     
                        tmp_fl = fl[0, :, :, :].cpu().numpy().astype(np.uint8)
                        tmp_fl_gland = tmp_fl[1,:,:]

                        comparison = tmp_fl_gland == w_label_m_to_f
                        tp = np.sum(np.logical_and(comparison, tmp_fl_gland == 1))
                        fn = np.sum(tmp_fl_gland == 1) - tp
                        fp = np.sum(w_label_m_to_f == 1) - tp
                        iou = tp/(fn+tp+fp)
                        recall = tp/(tp+fn)
                        dice = 2*tp / (2*tp + fn + fp)
                        reg_iou_list.append(iou)
                        reg_recall_list.append(recall)
                        reg_dice_list.append(dice)

                    if(self.is_save_flow == True):
                        flow = np.squeeze(flow) 

                        np.save(join(self.results_dir, self.model_name, 'flow', f'{name1}_{name2}.npy'), flow)

                pbar.update(mi.shape[0])

        reg_iou_average=sum(reg_iou_list)/len(reg_iou_list)
        reg_iou_std = np.std(reg_iou_list)


        reg_recall_average=sum(reg_recall_list)/len(reg_recall_list)
        reg_recall_std = np.std(reg_recall_list)

        reg_dice_average=sum(reg_dice_list)/len(reg_dice_list)
        reg_dice_std = np.std(reg_dice_list)


        return reg_iou_average+reg_recall_average+reg_dice_average
            
    def checkpoint_reg(self, epoch, k):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:
            Reg_model_path = os.path.join(self.checkpoint_dir,'Reg_Model')

            if not os.path.exists(Reg_model_path):
                os.makedirs(Reg_model_path)
            torch.save(self.Reger.state_dict(),
                        '{0}/{1}_epoch_{2}.pth'.format(Reg_model_path,'Reger_'+self.model_name.split("/")[0], epoch+k),
                        _use_new_zipfile_serialization=False)

    def train(self):
        max_reg_value = 0  
        best_reg_model_state_dict = self.Reger.state_dict()
        best_seg_epoch = -1
        best_reg_epoch = -1
        k = self.k
        self.train_info_list = [] 
        self.val_info_list = [] 
        
        if(self.pretrainRegmodel!=''): 
            self.Reger.load_state_dict(torch.load(self.pretrainRegmodel))
            best_seg_model_state_dict = self.Reger.state_dict()
        else:
            reg_best_count = 0
            best_reg_model_file_path = os.path.join(self.checkpoint_dir,'Best_Reg_Model')
            if not os.path.exists(best_reg_model_file_path):
                os.makedirs(best_reg_model_file_path)
            for epoch in range(self.epoches-self.k):
                self.L_smooth_log.reset()
                self.L_sim_log.reset()
                self.L_sim_2_log.reset()
                self.L_Reg_log.reset()
                before_weights = {}
                for name, param in self.Reger.named_parameters():
                    before_weights[name] = param.data.clone()               
                self.Reger.train()
                self.train_reg_epoch(epoch+self.k)
                self.Reger.eval()
                after_weights = {}
                for name, param in self.Reger.named_parameters():
                    after_weights[name] = param.data
                if epoch<100:
                    if epoch % 50 == 0:
                        self.checkpoint_reg(epoch, self.k)
                else :
                    if epoch % 10 == 0:
                        self.checkpoint_reg(epoch, self.k)
                    
                self.writer.add_scalar('Train/L_smooth_log', self.L_smooth_log.avg.cpu().item(), epoch)
                if(self.cal_ssim_rate!=0):
                    self.writer.add_scalar('Train/L_sim_log', self.L_sim_log.avg.cpu().item(), epoch)
                if(self.cal_ncc_rate!=0):
                    self.writer.add_scalar('Train/L_sim_2_log', self.L_sim_2_log.avg.cpu().item(), epoch)
                self.writer.add_scalar('Train/L_Reg_log', self.L_Reg_log.avg.cpu().item(), epoch)
                
                new_reg_value = self.test_reg(epoch)
                if new_reg_value > max_reg_value:
                    max_reg_value = new_reg_value
                    best_reg_epoch = epoch
                    best_reg_model_state_dict = self.Reger.state_dict()
                    reg_best_count+=1
                    torch.save(best_reg_model_state_dict,
                        '{0}/{1}_epoch_{2}.pth'.format(best_reg_model_file_path, 'Best_Reger_' + self.model_name.split("/")[0], best_reg_epoch + k),
                        _use_new_zipfile_serialization=False)

                    if(reg_best_count > self.save_best_mode_number):
                        delete_except_last_n(best_reg_model_file_path,self.save_best_mode_number)
                self.writer.add_scalar('Temp_Value_sum/reg_validate', new_reg_value, epoch)
                self.writer.add_scalar('Best_Value_sum/reg_validate', max_reg_value, epoch)

def set_random_seed(seed_value):
    """Set the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

 
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(3407)
    for fold in range(1, 6):
        print('the data cross: %s\n' % (str(fold)))
        with_fix_json_path="/mnt/xxx/Datasets/MGD/for_5fold/train_dict{}.json".format(fold)
        RegNet = Reg_MGD(device=device,with_fix_json_path=with_fix_json_path,foldnum=fold)
        RegNet.train()





