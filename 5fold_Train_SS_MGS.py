import glob
import medpy
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
from os.path import join
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from  PIL import Image
from models.Reg_network import UNet_reg_with_se_2pam as UNet_Reg
from models.Seg_network import UNet_MGS
from utils.STN import SpatialTransformer_2d, Re_SpatialTransformer_2d
from utils.dataloader_5fold_segmentation import Dataset_train_seg_25percent,Dataset_seg_test
from utils.losses import DiceCELoss
from utils.utils import AverageMeter,delete_except_last_n
import json
from tqdm import tqdm
import os
import shutil
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.modules import RNSF_ContrastiveLoss,TensorBuffer
import medpy.metric
class SS_MGS(object):
    def __init__(self,
                foldnum,
                k=0,
                n_channels=1,
                n_classes=2,
                lr=1e-3,
                epoches=50,
                iters=200,
                batch_size=1,
                test_batch_size=1,
                subject_num = -1,
                data_dir = 'your data file dir',
                datajsonpath = '',
                with_fix_json_path = '',
                checkpoint_dir='checkpoin_output_dir',
                result_dir='result_output_dir',
                is_save_flow = False,
                datajson_set_atlas = True,
                is_save_val_img = False,
                pretrainRegmodel='the address of the registration model.',
                train_seg = True,
                save_best_mode_number = 10, # Save the top n best models
                device = 0, 
                sample_num = 250, # Parameter K of the RSNF module
                buffer_size=1,
                is_mixed=True,# Mixed-precision training
                reg_supervised_rate = 0.5,# L_rs
                contrastive_rate = 0.1, # L_rnsf
                rampup_length = 50, 
                 ):#
        super(SS_MGS, self).__init__()
        # deep supervision output list 
        self.ds_list = [ 'level2','out']
        self.rampup_length = rampup_length
        self.reg_supervised_rate= reg_supervised_rate
        self.contrastive_rate=contrastive_rate
        # Create a gradient scaler for mixed-precision training
        self.scaler = GradScaler()
        self.is_mixed=is_mixed
        self.buffer_size=buffer_size
        self.sample_num=sample_num
        
         # Prepare a tensor buffer for contrastive learning if contrastive loss is used
        if(self.contrastive_rate!=0):
            self.prepare_tensor_buffer()
            self.contrastive_loss = RNSF_ContrastiveLoss(sample_num=self.sample_num, temperature=0.1)
        self.train_seg = train_seg
        self.foldnum=foldnum
        self.save_best_mode_number = save_best_mode_number
        # Find the best pre-trained registration model
        pth_files = glob.glob(os.path.join(pretrainRegmodel,str(fold), '*.pth')) 
        self.pretrainRegmodel=pth_files[0]
        # initialize parameters
        self.is_save_val_img = is_save_val_img
        self.is_save_flow =is_save_flow
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.batch_size =batch_size
        self.iters = iters
        self.lr = lr
        self.data_dir = data_dir
        self.datajsonpath = datajsonpath
        self.device=device
        self.results_dir = result_dir                                                                                        
       
        # Generate the model name
        self.model_name = 'The experiment name of the segmentation model'
        self.model_name =self.model_name + '/{}/'.format(self.foldnum)
        self.checkpoint_dir = os.path.join(checkpoint_dir,self.model_name)
        self.test_batch_size = test_batch_size
        self.with_fix_json_path = with_fix_json_path
        
        self.subject_num = subject_num

        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir= self.results_dir+'/'+self.model_name+'/logs')
        
        # Create necessary directories
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # Copy the current script to the checkpoint directory
        current_script_path = os.path.abspath(__file__)
        shutil.copy2(current_script_path, self.checkpoint_dir)

        # tools
        self.stn = SpatialTransformer_2d() # Spatial Transformer
        self.lstn = SpatialTransformer_2d()
        self.rstn = Re_SpatialTransformer_2d() # Spatial Transformer-inverse
        self.rlstn = Re_SpatialTransformer_2d()
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize networks
        self.Reger = UNet_Reg(n_channels=n_channels)
        self.Seger = UNet_MGS(n_channels=n_channels, n_classes=n_classes)

        # Move models to GPU if available
        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()


        # initialize optimizer
        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr,weight_decay=1e-6)
        self.optS = torch.optim.AdamW(self.Seger.parameters(), lr=lr,weight_decay=1e-6)

        # Set up the learning rate scheduler
        T_max = 100  # Maximum steps in one cycle
        self.scheduler = CosineAnnealingLR(self.optS, T_max=T_max, eta_min=0.0001)

        # Read the JSON file
        with open(self.datajsonpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Initialize data loaders
        train_dataset = Dataset_train_seg_25percent(self.data_dir,data["train"], self.n_classes, subject_num = self.subject_num,with_fix=datajson_set_atlas)
        self.dataloader_train = DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
        test_dataset_seg = Dataset_seg_test(self.data_dir,data["val"], self.n_classes)
        self.dataloader_val_seg = DataLoader(test_dataset_seg, batch_size=self.test_batch_size)

        # Define loss functions
        pos_weight = [1.0] 
        self.criterion = DiceCELoss(lambda_ce=0.5,weight=torch.tensor([1] + pos_weight).to(device))
        
        # Define loss logs
        self.L_Seg_log = AverageMeter(name='L_Seg')  # Overall loss
        self.L_seg_log = AverageMeter(name='L_seg') # Lseg
        self.L_contrastive_log = AverageMeter(name='L_contrastive') # L_rnsf
        self.L_reg_supervised1_log = AverageMeter(name='L_reg_supervised1') # L_rs1
        self.L_reg_supervised2_log = AverageMeter(name='L_reg_supervised2') # L_rs2


    def sigmoid_rampup(self, current):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        rampup_length = self.rampup_length
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


    def train_seg_iterator(self,labed_img, labed_lab, unlabed_img1, unlabed_img2,\
                l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,\
                l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,\
                epoch_num,iter_num,negative_img):
        
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to True below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = False  # they are set to False 
        
        if torch.cuda.is_available():
            l_u1_displace_field = l_u1_displace_field.cuda()
            l_u2_displace_field = l_u2_displace_field.cuda()
            u1_l_displace_field = u1_l_displace_field.cuda()
            u2_l_displace_field = u2_l_displace_field.cuda()
            u1_u2_displace_field = u1_u2_displace_field.cuda()
            u2_u1_displace_field = u2_u1_displace_field.cuda()            
            l_u1_img = l_u1_img.cuda()
            l_u2_img = l_u2_img.cuda()
            u1_l_img = u1_l_img.cuda()
            u2_l_img = u2_l_img.cuda()
            u1_u2_img =u1_u2_img.cuda()
            u2_u1_img =u2_u1_img.cuda()
            l_u1_seg =l_u1_seg.cuda()
            l_u2_seg = l_u2_seg.cuda()           
            
        with torch.no_grad():

            # RSGM
            w_u_to_l,_, _, _,_ = self.Reger(u1_l_img, labed_img, None, labed_lab) 
            _, _,_,_, flow2 = self.Reger(l_u2_img, unlabed_img2, l_u2_seg, None)

            beta = np.random.uniform(0, 1) 
            alpha = np.random.uniform(0, 1) 
            sty = beta * (w_u_to_l - labed_img)

            spa = alpha * (flow2+l_u2_displace_field)
            g2 = self.stn(labed_img + sty, spa) # g2
            g1 = self.stn(labed_img, spa)  # g1
            yg = self.stn(labed_lab, spa) #yg

            
        with autocast(enabled=self.is_mixed):  
            # Set input images and labels          
            self.image_l = labed_img
            self.label_l = labed_lab
            self.image_l_g1 = g1 
            self.image_l_g1_reg_lab = yg
            self.image_l_g2 = g2
            self.image_l_g2_reg_lab = yg
            self.image_u = negative_img
            inputs = torch.cat([
                self.image_l, self.image_l_g1, self.image_l_g2,self.image_u], dim=0)
                
            outputs = self.Seger(inputs)  
            # Deallocate batch outputs
            self.out_l = self.deallocate_batch_dict(outputs, 0, self.batch_size)
            self.out_l_g1 = self.deallocate_batch_dict(outputs, self.batch_size, 2 * self.batch_size)
            self.out_l_g2 = self.deallocate_batch_dict(outputs, 2 * self.batch_size, 3 * self.batch_size) 
            self.out_u = self.deallocate_batch_dict(outputs, 3 * self.batch_size, 4 * self.batch_size)
            
            # define predictions
            self.pred_l = self.out_l['out']  
            self.pred_g1 = self.out_l_g1['out'] 
            self.pred_g2 = self.out_l_g2['out']
            self.pred_u = self.out_u['out']  

            # Define projector variables for contrastive learning
            if(self.contrastive_rate!=0):
                self.project_l_g1 = self.out_l_g1['project']  
                self.project_l_g2 = self.out_l_g2['project']  
                self.project_l_negative.update(self.out_u['project'])  

                self.map_l_positive=self.out_l_g1['project_map']

                self.map_l_negative.update(self.out_u['project_map']) 

            # # Calculate deep supervision segmentation loss (L_seg)
            self.seg_loss = self.get_multi_loss(self.out_l, self.label_l, is_ds=True, key_list=self.ds_list)
            
            
            # Calculate L_rs
            if(self.reg_supervised_rate):

                self.rs_out_l_t1_loss = self.reg_supervised_loss(self.pred_g1,self.image_l_g1_reg_lab) 
                self.rs_out_l_t2_loss = self.reg_supervised_loss(self.pred_g2,self.image_l_g2_reg_lab) 
            
            # Calculate contrastive loss
            if(self.contrastive_rate!=0):
                self.contrastive_l_loss = torch.utils.checkpoint.checkpoint(
                    self.contrastive_loss, self.project_l_g1, self.project_l_g2, self.project_l_negative.values,
                    self.map_l_positive, self.map_l_negative.values)
            tau = self.sigmoid_rampup(epoch_num)
            
            # Initialize loss components
            seg_loss = self.seg_loss
            contrastive_l_loss=0
            reg_supervised1=0
            reg_supervised2=0
            # Update loss components
            if(self.contrastive_rate!=0):
                contrastive_l_loss = self.contrastive_rate * tau * (self.contrastive_l_loss)
                self.L_contrastive_log.update(contrastive_l_loss.data, self.pred_l.size(0))
            if(self.reg_supervised_rate):
                a = 1
                reg_supervised1 = self.reg_supervised_rate*self.rs_out_l_t1_loss*a
                reg_supervised2 = self.reg_supervised_rate*self.rs_out_l_t2_loss*a
                
                self.L_reg_supervised1_log.update(reg_supervised1.data, self.pred_l.size(0))
                self.L_reg_supervised2_log.update(reg_supervised2.data, self.pred_l.size(0))
            
            # Calculate final loss
            final_loss = seg_loss + contrastive_l_loss+reg_supervised1+reg_supervised2
            
            # Update loss logs
            self.L_seg_log.update(seg_loss.data, self.pred_l.size(0))
            self.L_Seg_log.update(final_loss.data, self.pred_l.size(0))
                             
        # Backward pass and optimization                     
        self.optS.zero_grad()
        if self.is_mixed:
            self.scaler.scale(final_loss).backward()
            self.scaler.step(self.optS)
            self.scaler.update()
        else:
            final_loss.backward()
            self.optS.step()
            # 

    def consist_loss(self, inputs, targets, key_list=None):
        """
        Consistency regularization between two augmented views
        """
        loss = 0.0
        keys = key_list if key_list is not None else list(inputs.keys())
        for key in keys:
            loss += (1.0 - F.cosine_similarity(inputs[key], targets[key], dim=1)).mean()
        return loss            
            
    def reg_supervised_loss(self, inputs, targets):

        reg_supervised_loss = self.criterion(inputs,targets)
        return reg_supervised_loss               
        
           
    def dict_loss(self, loss_fn, inputs, targets, key_list=None, **kwargs):

        loss = 0.0
        keys = key_list if key_list is not None else list(inputs.keys())
        for key in keys:
            loss += loss_fn(inputs[key], targets, **kwargs)
        return loss

    def get_multi_loss(self, out_dict, label, is_ds=True, key_list=None):
        keys = key_list if key_list is not None else list(out_dict.keys())
        if is_ds:
            multi_loss = sum([self.criterion(out_dict[key], label) for key in keys])
        else:
            multi_loss = self.criterion(out_dict['out'], label)
        return multi_loss
    def deallocate_batch_dict(self, input_dict, batch_idx_start, batch_idx_end):
        """
        Deallocate the dict containing multiple batches into a dict with multiple items
        """
        out_dict = {}
        for key, value in input_dict.items():
            out_dict[key] = value[batch_idx_start:batch_idx_end]
        return out_dict        
                
    def train_seg_epoch(self, epoch):

        total_iters = self.iters 

        with tqdm(total=total_iters, desc='epoch', ncols=80) as pbar:
            for i in range(self.iters):
                labed_img, labed_lab, unlabed_img1, unlabed_img2,\
                l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,\
                l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,\
                u1_u2_displace_field,u2_u1_displace_field,negative_img= next(self.dataloader_train.__iter__())

                if torch.cuda.is_available():
                    labed_img = labed_img.cuda()
                    labed_lab = labed_lab.cuda()
                    unlabed_img1 = unlabed_img1.cuda()
                    unlabed_img2 = unlabed_img2.cuda()
                    negative_img = negative_img.cuda()

                self.train_seg_iterator(labed_img, labed_lab, unlabed_img1, unlabed_img2,\
                                    l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,\
                                    l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,\
                                    epoch+1,i,negative_img)
                
                
                res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                                'Iter: [%d/%d]' % (i + 1, self.iters),
                                self.L_seg_log.__str__(),
                                self.L_reg_supervised1_log.__str__(),
                                self.L_reg_supervised2_log.__str__(),
                                self.L_contrastive_log.__str__(),
                                self.L_Seg_log.__str__(),
                                ])

                print(res)
                pbar.update(1)
    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Seger(mi)
        return s_m['out']

    def val_seg(self,epoch):
        seg_iou_list = []
        seg_recall_list = []
        seg_dice_list = []
        seg_acc_list = []
        seg_namelist = []
        hd95_list = []

        self.Seger.eval()

        total_samples = len(self.dataloader_val_seg.dataset)  
        with tqdm(total=total_samples, desc='Testing Segmentation', ncols=80) as pbar:
            for i, (mi, ml, name) in enumerate(self.dataloader_val_seg):
                
                seg_namelist.append(name)
      
                name = name[0]
                if torch.cuda.is_available():
                    mi = mi.cuda()

                s_m = self.test_iterator_seg(mi) 
                s_m = np.argmax(s_m.data.cpu().numpy()[0], axis=0) 
                s_m = s_m.astype(np.int8) 

                s_m = np.squeeze(s_m)
                if(self.n_classes==2 ):

                    if(self.is_save_val_img == True):
                        if not os.path.exists(join(self.results_dir, self.model_name,str(self.foldnum), 'seg')):
                            os.makedirs(join(self.results_dir, self.model_name, str(self.foldnum),'seg'))
                    tmps_m = s_m.astype(np.uint8)
                    s_m = (s_m * 255).astype(np.uint8)
                    if(self.is_save_val_img == True):
                        s_m = Image.fromarray(s_m)
                        s_m.save(join(self.results_dir, self.model_name, str(self.foldnum),'seg', name + '.jpg'))


                    tmp_ml = ml[0, :, :, :].numpy().astype(np.uint8)
                    tmp_ml_gland = tmp_ml[1,:,:]

                    comparison = tmp_ml_gland == tmps_m
                    tp = np.sum(np.logical_and(comparison, tmp_ml_gland == 1))

                    fn = np.sum(tmp_ml_gland == 1) - tp
                    fp = np.sum(tmps_m == 1) - tp

                    tn = np.sum(np.logical_and(comparison, tmp_ml_gland == 0))
                    iou = tp/(fn+tp+fp)
                    recall = tp/(tp+fn)
                    dice = 2*tp / (2*tp + fn + fp)
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    seg_iou_list.append(iou)
                    seg_recall_list.append(recall)
                    seg_dice_list.append(dice)
                    seg_acc_list.append(accuracy)
                    hd95_list.append(medpy.metric.binary.hd95(tmps_m,tmp_ml_gland))
                pbar.update(mi.shape[0])


        seg_iou_average=sum(seg_iou_list)/len(seg_iou_list)
        seg_iou_std = np.std(seg_iou_list)


        seg_recall_average=sum(seg_recall_list)/len(seg_recall_list)
        seg_recall_std = np.std(seg_recall_list)


        seg_dice_average=sum(seg_dice_list)/len(seg_dice_list)
        seg_dice_std = np.std(seg_dice_list)

        
        seg_acc_average=sum(seg_acc_list)/len(seg_acc_list)
        seg_acc_std = np.std(seg_acc_list)
        
        hd95_average = sum(hd95_list)/len(hd95_list)
        hd95_std = np.std(hd95_list)
        return seg_iou_average,seg_recall_average,seg_dice_average,seg_acc_average,hd95_average

    def checkpoint(self, epoch, k):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:
            print(self.model_name.split("/")[0])
            torch.save(self.Seger.state_dict(),
                        '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_'+self.model_name.split("/")[0], epoch+k),
                        _use_new_zipfile_serialization=False)
            torch.save(self.Reger.state_dict(),
                        '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name.split("/")[0], epoch+k),
                        _use_new_zipfile_serialization=False)

    def load(self,Best_Seger_path = '' ,Best_Reger_path=''):
        if(Best_Reger_path=='' and Best_Seger_path == '' ):
            self.Reger.load_state_dict(
                torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name.split("/")[0], str(self.k))))
            self.Seger.load_state_dict(
                torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_' + self.model_name.split("/")[0], str(self.k))))

        else:
            self.Seger.load_state_dict(torch.load(Best_Seger_path))       
            self.Reger.load_state_dict(torch.load(Best_Reger_path))
    def checkpoint_seg(self, epoch, k):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:

            Seg_model_path =  os.path.join(self.checkpoint_dir,'Seg_Model')

            if not os.path.exists(Seg_model_path):
                os.makedirs(Seg_model_path)
                
            torch.save(self.Seger.state_dict(),
                        '{0}/{1}_epoch_{2}.pth'.format(Seg_model_path, 'Seger_'+self.model_name.split("/")[0], epoch+k),
                        _use_new_zipfile_serialization=False)
            
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

        max_seg_value = 0  
        max_reg_value = 0  
        
        best_seg_model_state_dict = self.Seger.state_dict()
        best_reg_model_state_dict = self.Reger.state_dict()
        best_seg_epoch = -1
        best_reg_epoch = -1
        k = self.k
        self.train_info_list = [] 
        self.val_info_list = [] 
        
        if(self.pretrainRegmodel!=''): 
            self.Reger.load_state_dict(torch.load(self.pretrainRegmodel))
            best_reg_model_state_dict = self.Reger.state_dict()
        else:
            print("need a pretrain reg model")
        if (self.train_seg == True):

            seg_best_count = 0
            best_seg_model_file_path = os.path.join(self.checkpoint_dir,'Best_Seg_Model')
            if not os.path.exists(best_seg_model_file_path):
                os.makedirs(best_seg_model_file_path)
            if not os.path.exists(os.path.join(self.checkpoint_dir,'Seg_Model')):
                os.makedirs(os.path.join(self.checkpoint_dir,'Seg_Model'))
            if not os.path.exists(best_seg_model_file_path):
                os.makedirs(best_seg_model_file_path)
            for epoch in range(self.epoches-self.k):
                self.L_Seg_log.reset()
                self.L_contrastive_log.reset()
                self.L_reg_supervised1_log.reset()
                self.L_reg_supervised2_log.reset()
                self.L_seg_log.reset()
                self.Seger.train()

                self.Reger.eval()
                self.train_seg_epoch(epoch+self.k)

                self.scheduler.step()
                
                self.Seger.eval()


                if(self.epoches>100):
                    if epoch<100:
                        if epoch % 50 == 0:
                            self.checkpoint_seg(epoch, self.k)
                    else :
                        if epoch % 10 == 0:
                            self.checkpoint_seg(epoch, self.k)
                elif(self.epoches<=50):
                    self.checkpoint_seg(epoch, self.k)
                else:
                    if epoch % 10 == 0:
                        self.checkpoint_seg(epoch, self.k)
                
        
                self.writer.add_scalar('Train/L_seg_log', self.L_seg_log.avg.cpu().item(), epoch)
                self.writer.add_scalar('Train/L_Seg_log', self.L_Seg_log.avg.cpu().item(), epoch)
                if(self.contrastive_rate!=0):
                    self.writer.add_scalar('Train/L_contrastive_log', self.L_contrastive_log.avg.cpu().item(), epoch)
                if(self.reg_supervised_rate):
                    self.writer.add_scalar('Train/L_reg_supervised1_log', self.L_reg_supervised1_log.avg.cpu().item(), epoch)
                    self.writer.add_scalar('Train/L_reg_supervised2_log', self.L_reg_supervised2_log.avg.cpu().item(), epoch)
                    
                seg_iou_average,seg_recall_average,seg_dice_average,seg_acc_average,hd_average= self.val_seg(epoch)
                new_seg_value=seg_iou_average
                if new_seg_value > max_seg_value:
                    max_seg_value = new_seg_value
                    best_seg_epoch = epoch
                    seg_best_count+=1
                    best_seg_model_state_dict = self.Seger.state_dict()

                    torch.save(best_seg_model_state_dict,
                            '{0}/{1}_epoch_{2}.pth'.format(best_seg_model_file_path, 'Best_Seger_' + self.model_name.split("/")[0], best_seg_epoch + k),
                            _use_new_zipfile_serialization=False)
                    if(seg_best_count>self.save_best_mode_number):
                        delete_except_last_n(best_seg_model_file_path,self.save_best_mode_number)

                self.writer.add_scalar('Temp_Value_sum/seg_dice_validate', seg_dice_average, epoch)
                self.writer.add_scalar('Temp_Value_sum/seg_iou_validate', seg_iou_average, epoch)
                self.writer.add_scalar('Temp_Value_sum/seg_recall_validate', seg_recall_average, epoch)
                self.writer.add_scalar('Temp_Value_sum/seg_acc_validate', seg_acc_average, epoch)
                self.writer.add_scalar('Temp_Value_sum/hd95_validate', hd_average, epoch)                
                self.writer.add_scalar('Best_Value_sum/seg_validate', max_seg_value, epoch)
            
        self.writer.close()
            
        xlsl_path = os.path.join(self.results_dir , self.model_name)
        if not os.path.exists(xlsl_path):
            os.makedirs(xlsl_path)

    def prepare_tensor_buffer(self):
        self.project_l_negative = TensorBuffer(buffer_size=self.buffer_size, concat_dim=0)
        self.map_l_negative = TensorBuffer(buffer_size=self.buffer_size, concat_dim=0)


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
        MGSNet = SS_MGS(device=device,with_fix_json_path=with_fix_json_path,foldnum=fold)
        MGSNet.train()





