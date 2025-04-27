import os
from os.path import join
import random
import shutil
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from  PIL import Image
from models.Reg_network import UNet_reg_with_se_2pam as UNet_Reg
from models.Seg_network import UNet_MGS
from utils.dataloader_5fold_registration import Dataset_reg_test 
from utils.dataloader_5fold_segmentation import Dataset_seg_test
from utils.save_loss import save_val_info_to_excel_add_acc,save_val_info_to_excel_add_acc_hd95
import json
from tqdm import tqdm
import cv2
import medpy.metric
import medpy
import medpy.metric
from utils.utils import  draw_gland_over_image_and_fix_moving_fixed_imgs_without_mask
    
def draw_gland_over_image(save_path, gland_image, mask_image, overlay_color, alpha):
    gland_image_color = Image.fromarray(cv2.cvtColor(gland_image, cv2.COLOR_GRAY2BGR))
    overlay_image = np.full_like(np.array(gland_image_color), overlay_color, dtype=np.uint8)
    overlay_image_pil = Image.fromarray(overlay_image)
    overlayed_channels = []
    for channel in overlay_image_pil.split():
        overlayed_channel = Image.fromarray(np.bitwise_and(np.array(channel), mask_image))
        overlayed_channels.append(overlayed_channel)
    overlayed_image = Image.merge('RGB', overlayed_channels)
    result_image = Image.blend(gland_image_color, overlayed_image, alpha)
    result_image.save(save_path)

class Reg_Seg_Test(object):
    def __init__(self,foldnum,
                n_channels=1,
                n_classes=2,
                test_batch_size=1,
                subject_num = -1,
                data_dir = '/mnt/XX/Datasets/MGD',
                with_fix = True,
                datajsonpath = '',
                with_fix_json_path = '',
                checkpoint_dir='checkpoin_output_dir',
                result_dir='result_output_dir',
                is_save_flow = False,
                is_save_test_img = False,
                device = 0,
                img_mode = 'bilinear',
                seg_mode = 'nearest',
                model_name = '',
                 ):#
        
        super(Reg_Seg_Test, self).__init__()
        self.test_batch_size = test_batch_size
        self.foldnum=foldnum
        self.device=device
        # initialize parameters
        self.is_save_test_img = is_save_test_img
        self.is_save_flow =is_save_flow
        self.n_classes = n_classes
        self.data_dir = data_dir
        self.datajsonpath = datajsonpath
        self.results_dir = result_dir
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(checkpoint_dir,self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        self.img_mode = img_mode
        self.seg_mode = seg_mode
        self.with_fix =with_fix
        self.with_fix_json_path = with_fix_json_path
        
        if self.with_fix:
            self.datajsonpath = self.with_fix_json_path
            
        self.subject_num = subject_num
        self.softmax = nn.Softmax(dim=1)

        self.Reger = UNet_Reg(n_channels=n_channels)
        self.Seger = UNet_MGS(n_channels=n_channels, n_classes=n_classes)
        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()
        with open(self.datajsonpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        test_dataset_seg = Dataset_seg_test(self.data_dir,data["test"], self.n_classes ,with_fix=self.with_fix)
        self.dataloader_test_seg = DataLoader(test_dataset_seg, batch_size=self.test_batch_size)
        test_dataset_reg = Dataset_reg_test(self.data_dir,data["test"], self.n_classes,with_fix=self.with_fix)
        self.dataloader_test_reg = DataLoader(test_dataset_reg, batch_size=self.test_batch_size)

    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Seger(mi)
        return s_m['out']

    def test_iterator_reg(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        return w_m_to_f, w_label_m_to_f, flow

    def for_many_test_reg(self, result_dir):
        self.results_dir = result_dir
        reg_iou_list = []
        reg_recall_list = []
        reg_dice_list = []
        self.Seger.eval()
        self.Reger.eval()
        val_info_list = [] #

        total2_samples = len(self.dataloader_test_reg.dataset)  
        with tqdm(total=total2_samples, desc='Testing Regression', ncols=80) as pbar:
            for i, (mi, ml, fi, fl, l1_l2_img,l2_l1_img,l1_l2_seg,l2_l1_seg,l1_l2_displace_field,l2_l1_displace_field,name1, name2) in enumerate(self.dataloader_test_reg):
                name1 = name1[0] 
                name2 = name2[0] 
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
                    if(self.is_save_flow == True and not os.path.exists(join(self.results_dir, self.model_name, str(self.foldnum),'flow'))):
                        os.makedirs(join(self.results_dir, self.model_name, str(self.foldnum),'flow'))
                    if(self.is_save_test_img == True and not os.path.exists(join(self.results_dir, self.model_name, str(self.foldnum),'w_m_to_f'))):
                        os.makedirs(join(self.results_dir, self.model_name,str(self.foldnum), 'w_m_to_f'))

                    w_m_to_f = np.squeeze(w_m_to_f)
                    w_m_to_f = (w_m_to_f * 255).astype(np.uint8)
                    w_m_to_f_tmp = Image.fromarray(w_m_to_f)

                    if(self.is_save_test_img == True):
                        w_m_to_f_tmp.save(
                            join(self.results_dir, self.model_name,str(self.foldnum), 'w_m_to_f', name1 + '_' + name2 + '.jpg'))

                    if(self.n_classes ==2 ):
                        if not os.path.exists(join(self.results_dir, self.model_name,str(self.foldnum), 'w_label_m_to_f')):
                            os.makedirs(join(self.results_dir, self.model_name,str(self.foldnum), 'w_label_m_to_f'))
                        w_label_m_to_f = np.squeeze(w_label_m_to_f)
                        w_label_m_to_f = (w_label_m_to_f * 255).astype(np.uint8)
                        if( self.is_save_test_img == True):
                            w_label_m_to_f_image = Image.fromarray(w_label_m_to_f)
                            w_label_m_to_f_image.save(
                                join(self.results_dir, self.model_name,str(self.foldnum), 'w_label_m_to_f', name1 + '_' + name2 + '.jpg'))
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
                
                        np.save(join(self.results_dir, self.model_name, str(self.foldnum),'flow', name1[:-4] + '_' + name2[:-4] + '.npy'), flow)

                pbar.update(mi.shape[0])
                
                if(self.is_save_test_img):
                    if(self.n_classes==2):
                        plt.rcParams['font.size'] = 45
                        fig, axes =plt.subplots(nrows=2,ncols=5,figsize=(50, 40))
                        tmp_moveimg = (np.squeeze(mi).cpu().numpy())*255
                        axes[0,0].imshow(tmp_moveimg.astype(np.uint8),cmap="gray")
                        axes[0,0].set_title("Moved Image")

                        tmp_movlabel = (ml[0, 1,:,:].cpu().numpy())*255
                        axes[1,0].imshow(tmp_movlabel.astype(np.uint8),cmap="gray")
                        axes[1,0].set_title("Moved Label")
                        tmp_fixedimg = (fi[0, 0 , :, :].cpu().numpy())*255
                        axes[0,1].imshow(tmp_fixedimg.astype(np.uint8),cmap="gray")
                        axes[0,1].set_title("Fixed Image")

                        tmp_fixedlabel = (fl[0, 1, :, :].cpu().numpy())*255
                        axes[1,1].imshow(tmp_fixedlabel.astype(np.uint8),cmap="gray")
                        axes[1,1].set_title("Fixed Label")
                        
                        l1_l2_imgtemp = (np.squeeze(l1_l2_img).cpu().numpy())*255
                        axes[0,2].imshow(l1_l2_imgtemp.astype(np.uint8),cmap="gray")
                        axes[0,2].set_title("Affine Image")
                        
                        l1_l2_segtemp = (np.squeeze(l1_l2_seg[0,1,:,:]).cpu().numpy())*255
                        axes[1,2].imshow(l1_l2_segtemp.astype(np.uint8),cmap="gray")
                        axes[1,2].set_title("Affine gland")

                        axes[0,3].imshow((w_m_to_f).astype(np.uint8),cmap="gray")
                        axes[0,3].set_title("Warped Image")
                        
                        tmp_w_gland_label_m_to_f = w_label_m_to_f

                        axes[1,3].imshow(tmp_w_gland_label_m_to_f.astype(np.uint8),cmap="gray")
                        axes[1,3].set_title("Warped Gland")

                        overlay_color = [(128, 0, 128),(255,0,255)] 
                        alpha = 0.5
                        save_path = os.path.join(self.results_dir,self.model_name+'/'+str(self.foldnum)+'/compair_registration')
                        f_grand_label = (fl[0, 1, :, :].cpu().numpy())*255
                        f_label = f_grand_label
                        overlayimg = draw_gland_over_image_and_fix_moving_fixed_imgs_without_mask(save_path,(w_m_to_f).astype(np.uint8),tmp_w_gland_label_m_to_f.astype(np.uint8),tmp_fixedimg.astype(np.uint8),f_label.astype(np.uint8),name1,name2,overlay_color,alpha)
                        axes[0,4].imshow(overlayimg)
                        axes[0,4].set_title("Overlay Img")

                        plt.tight_layout()
                        plt.axis('off')
                        save_generate_image_file = os.path.join(self.results_dir,self.model_name+'/'+str(self.foldnum)+"/all_register_image_result")
                        if not os.path.exists(save_generate_image_file):
                            os.makedirs(save_generate_image_file)
                        generate_image_name = name1 + '_' + name2 + '.jpg'

                        plt.savefig(os.path.join(save_generate_image_file,generate_image_name))

        reg_iou_average=sum(reg_iou_list)/len(reg_iou_list)
        reg_iou_std = np.std(reg_iou_list)


        reg_recall_average=sum(reg_recall_list)/len(reg_recall_list)
        reg_recall_std = np.std(reg_recall_list)


        reg_dice_average=sum(reg_dice_list)/len(reg_dice_list)
        reg_dice_std = np.std(reg_dice_list)

        val_info_list.append([0,0,0,
                                    0,0,
                                    0,0,
                                    0,0,
                                    reg_iou_average,reg_iou_std,
                                    reg_recall_average,reg_recall_std,
                                    reg_dice_average,reg_dice_std])
            
        xlsl_path = os.path.join(self.results_dir , self.model_name)
        if not os.path.exists(xlsl_path):
            os.makedirs(xlsl_path)
        save_val_info_to_excel_add_acc(val_info_list,os.path.join(xlsl_path,'test_reg_info.xlsx'))

        return reg_iou_average,reg_recall_average,reg_dice_average
    def for_many_test_seg(self, result_dir):
        self.results_dir = result_dir
        seg_iou_list = []
        seg_recall_list = []
        seg_dice_list = []
        seg_namelist = []
        seg_acc_list = []
        self.Seger.eval()
        val_info_list =[]
        hd95_list = []

        total_samples = len(self.dataloader_test_seg.dataset)  
        with tqdm(total=total_samples, desc='Testing Segmentation', ncols=80) as pbar:
            for i, (mi, ml, name) in enumerate(self.dataloader_test_seg):
                
                seg_namelist.append(name)

                name = name[0]
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    
                s_m = self.test_iterator_seg(mi)
                s_m = np.argmax(s_m.data.cpu().numpy()[0], axis=0)
                s_m = s_m.astype(np.int8) 

                s_m = np.squeeze(s_m)
                orignal_s_m = s_m
                if(self.n_classes==2 ):
                    if(self.is_save_test_img == True):
                        if not os.path.exists(join(self.results_dir, self.model_name,str(self.foldnum), 'seg')):
                            os.makedirs(join(self.results_dir, self.model_name, str(self.foldnum),'seg'))
                    tmps_m = s_m.astype(np.uint8)
                    s_m = (s_m * 255).astype(np.uint8)
                    if(self.is_save_test_img == True):
                        s_m = Image.fromarray(s_m)
                        s_m.save(join(self.results_dir, self.model_name,str(self.foldnum), 'seg', name[:-4] + '.jpg'))
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
                    hd95_value = medpy.metric.binary.hd95(tmps_m, tmp_ml_gland)
                    hd95_list.append(hd95_value)

                if(self.is_save_test_img):
                    
                    generate_image_name = name[:-4] +"_with_gland_label.jpg"
                    save_generate_image_file = os.path.join(self.results_dir,self.model_name+'/'+str(self.foldnum)+"/all_segmentation_image_result")
                    if not os.path.exists(save_generate_image_file):
                            os.makedirs(save_generate_image_file)
                    save_path = os.path.join(save_generate_image_file,generate_image_name)
                    if(self.n_classes==2):
                        tmpgland = tmps_m
                    tmpgland = np.where(tmpgland==1,255,tmpgland)
                    gland_image =  ((np.squeeze(mi).cpu().numpy())*255).astype(np.uint8)

                    overlay_color = (255, 100, 100)
                    alpha = 0.2 
                    draw_gland_over_image(save_path, gland_image, tmpgland, overlay_color, alpha)

                    file_path = name[:-4] +"_compair.jpg"
                    save_compair_image_file = os.path.join(self.results_dir,self.model_name+'/'+str(self.foldnum)+"/compair_segmentation_image_result")
                    if not os.path.exists(save_compair_image_file):
                        os.makedirs(save_compair_image_file)
                    input_img = gland_image
                    lab = (tmp_ml_gland*255).astype(np.uint8)
                    input_img = cv2.resize(input_img,(tmpgland.shape[1], tmpgland.shape[0]))
                    pred_img = tmpgland
                    lab_img = lab
                    img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                    img[:, :input_img.shape[1]] = input_img
                    img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                    img[:, input_img.shape[1] * 2:] = lab_img
                    imgt = Image.fromarray(img.astype(np.uint8))
                    imgt.save(join(save_compair_image_file,file_path))
                pbar.update(mi.shape[0])

        seg_iou_average=sum(seg_iou_list)/len(seg_iou_list)
        seg_iou_std = np.std(seg_iou_list)


        seg_recall_average=sum(seg_recall_list)/len(seg_recall_list)
        seg_recall_std = np.std(seg_recall_list)


        seg_dice_average=sum(seg_dice_list)/len(seg_dice_list)
        seg_dice_std = np.std(seg_dice_list)

        
        seg_acc_average=sum(seg_acc_list)/len(seg_acc_list)
        seg_acc_std = np.std(seg_acc_list)
        if(len(hd95_list)!=0):
            hd95_average=sum(hd95_list)/len(hd95_list)
            hd95_std = np.std(hd95_list)
        else:
            hd95_average=0
            hd95_std=0

        val_info_list.append([0,seg_iou_average,seg_iou_std,
                                    seg_recall_average,seg_recall_std,
                                    seg_dice_average,seg_dice_std,
                                    seg_acc_average,seg_acc_std,
                                    hd95_average,hd95_std,
                                    0,0,
                                    0,0,
                                    0,0])
            
        xlsl_path = os.path.join(self.results_dir , self.model_name,str(self.foldnum))
        if not os.path.exists(xlsl_path):
            os.makedirs(xlsl_path)
        save_val_info_to_excel_add_acc_hd95(val_info_list,os.path.join(xlsl_path,'test_seg_info.xlsx'))
        
        return seg_iou_average,seg_recall_average,seg_dice_average,seg_acc_average,hd95_average

    def load_reg(self,Best_Reger_path=''):
        self.Reger.load_state_dict(torch.load(Best_Reger_path))      
    def load_seg(self,Best_Seger_path=''):
        self.Seger.load_state_dict(torch.load(Best_Seger_path))           
            
def set_random_seed(seed_value):
    """Set the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def list_files_in_directory(directory):

    if not os.path.isdir(directory):
        print(f"dir '{directory}' does not exist")
        return []

    filenames = os.listdir(directory)

    file_paths = [os.path.join(directory, filename) for filename in filenames if os.path.isfile(os.path.join(directory, filename))]

    return file_paths

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_manay_model=True
    test_reg = True
    test_seg = True
    set_random_seed(3407)
    seg_iou_list = []
    seg_recall_list = []
    seg_dice_list = []
    seg_acc_list = []
    seg_namelist = []
    reg_iou_list = []
    reg_recall_list = []
    reg_dice_list = []
    test_reg_info_list = []
    test_seg_info_list = []
    hd95_list=[]
    
    for fold in range(1, 6):
        
        print('the data cross: %s\n' % (str(fold)))
        with_fix_json_path="/mnt/xxx/Datasets/MGD/for_5fold/train_dict{}.json".format(fold)
        RSTNet = Reg_Seg_Test(device=device,with_fix_json_path=with_fix_json_path,foldnum=fold)
        test_result_dir = ""
        data_dir = '/mnt/xxx/Datasets/MGD'
        if(test_manay_model): 
            model_file = ''# The path of the model to be tested
            model_file = join(model_file,str(fold))
            with_fix = True 
            model_name = model_file.split("/")[-2]
            RSTNet = Reg_Seg_Test(device=device,with_fix_json_path=with_fix_json_path,foldnum=fold,result_dir=test_result_dir,data_dir = data_dir,model_name = model_name,with_fix = with_fix,is_save_test_img=False)
            
            reg_model_file = os.path.join(model_file,"Reg_Model")
            best_reg_model_file = os.path.join(model_file,"Best_Reg_Model")
            seg_model_file = os.path.join(model_file,"Seg_Model")
            best_seg_model_file = os.path.join(model_file,"Best_Seg_Model")        
            
            reg_model_list = list_files_in_directory(reg_model_file)
            reg_model_list = reg_model_list+list_files_in_directory(best_reg_model_file)
        
            seg_model_list = list_files_in_directory(seg_model_file)
            seg_model_list = seg_model_list+list_files_in_directory(best_seg_model_file)
            if(test_reg):
                best_reg_model_path =''
                max_reg_value = 0
                for reg_model_path in reg_model_list:
                    RSTNet.load_reg(reg_model_path)
                    print( os.path.basename(reg_model_path))
                    reg_iou_average,reg_recall_average,reg_dice_average =RSTNet.for_many_test_reg(result_dir=test_result_dir)
                    new_reg_value=reg_iou_average+reg_recall_average+reg_dice_average
                    if new_reg_value > max_reg_value:
                        max_reg_value = new_reg_value
                        best_reg_model_path = reg_model_path
                    print(new_reg_value)
                    
                print( os.path.basename(best_reg_model_path))
                print(max_reg_value)            
                RSTNet = Reg_Seg_Test(device=device,with_fix_json_path=with_fix_json_path,foldnum=fold,result_dir=test_result_dir,data_dir = data_dir,model_name = model_name,with_fix = with_fix,is_save_test_img=True)  
                RSTNet.load_reg(best_reg_model_path)  
                reg_iou_average,reg_recall_average,reg_dice_average=RSTNet.for_many_test_reg(result_dir=test_result_dir)
                reg_iou_list.append(reg_iou_average)
                reg_recall_list.append(reg_recall_average)
                reg_dice_list.append(reg_dice_average)
                destination_file = os.path.join(model_file, os.path.basename(best_reg_model_path))

                if not os.path.exists(best_reg_model_path):
                    print(f"Source file '{best_reg_model_path}' does not exist.")
                else:
                    if os.path.exists(destination_file):
                        os.remove(destination_file)
                    try:
                        shutil.copy2(best_reg_model_path, destination_file)
                        print(f"File copied successfully to {destination_file}")
                    except Exception as e:
                        print(f"Failed to copy file: {e}")
                
                head = os.path.basename(best_reg_model_path).split('_')[0]+os.path.basename(best_reg_model_path).split('_')[-1]
                test_reg_info_list.append([head,0,0,
                                0,0,
                                0,0,
                                0,0,
                                reg_iou_average,0,
                                reg_recall_average,0,
                                reg_dice_average,0])

            if(test_seg ):
                best_seg_model_path =''
                max_seg_value = 0
                for seg_model_path in seg_model_list:
                    RSTNet.load_seg(seg_model_path)
                    print( os.path.basename(seg_model_path))
                    
                    seg_iou_average,seg_recall_average,seg_dice_average,seg_acc_average,hd95_average=RSTNet.for_many_test_seg(result_dir=test_result_dir)
                    new_seg_value=seg_iou_average
                    if new_seg_value > max_seg_value:
                        max_seg_value = new_seg_value
                        best_seg_model_path = seg_model_path
                    print(new_seg_value)
                    
                print( os.path.basename(best_seg_model_path))
                print(max_seg_value)            
                RSTNet = Reg_Seg_Test(device=device,with_fix_json_path=with_fix_json_path,foldnum=fold,result_dir=test_result_dir,data_dir = data_dir,model_name = model_name,with_fix = with_fix,is_save_test_img=True)  
                RSTNet.load_seg(best_seg_model_path)  
                seg_iou_average,seg_recall_average,seg_dice_average,seg_acc_average,hd95_average=RSTNet.for_many_test_seg(result_dir=test_result_dir)
                seg_iou_list.append(seg_iou_average)
                seg_recall_list.append(seg_recall_average)
                seg_dice_list.append(seg_dice_average)
                seg_acc_list.append(seg_acc_average)
                hd95_list.append(hd95_average)
                destination_file = os.path.join(model_file, os.path.basename(best_seg_model_path))
                if not os.path.exists(best_seg_model_path):
                    print(f"Source file '{best_seg_model_path}' does not exist.")
                else:
                    if os.path.exists(destination_file):
                        os.remove(destination_file)

                    try:
                        shutil.copy2(best_seg_model_path, destination_file) 
                        print(f"File copied successfully to {destination_file}")
                    except Exception as e:
                        print(f"Failed to copy file: {e}")
                head = os.path.basename(best_seg_model_path).split('_')[0]+os.path.basename(best_seg_model_path).split('_')[-1]
                test_seg_info_list.append([head,seg_iou_average,0,
                                seg_recall_average,0,
                                seg_dice_average,0,
                                seg_acc_average,0,
                                hd95_average,0,
                                0,0,
                                0,0,
                                0,0])

    if(test_reg == True ):            
        reg_iou_average5fold=sum(reg_iou_list)/len(reg_iou_list)
        reg_iou_std5fold = np.std(reg_iou_list)


        reg_recall_average5fold=sum(reg_recall_list)/len(reg_recall_list)
        reg_recall_std5fold = np.std(reg_recall_list)


        reg_dice_average5fold=sum(reg_dice_list)/len(reg_dice_list)
        reg_dice_std5fold = np.std(reg_dice_list)

        test_reg_info_list.append(['all',0,0,
                                    0,0,
                                    0,0,
                                    0,0,
                                    reg_iou_average5fold,reg_iou_std5fold,
                                    reg_recall_average5fold,reg_recall_std5fold,
                                    reg_dice_average5fold,reg_dice_std5fold])
                                
        xlsl_path = os.path.join(test_result_dir, model_name)
        if not os.path.exists(xlsl_path):
            os.makedirs(xlsl_path)
        save_val_info_to_excel_add_acc(test_reg_info_list,os.path.join(xlsl_path,'test_reg_info.xlsx'))
    
    
    if(test_seg == True):
        seg_iou_average5fold=sum(seg_iou_list)/len(seg_iou_list)
        seg_iou_std5fold = np.std(seg_iou_list)


        seg_recall_average5fold=sum(seg_recall_list)/len(seg_recall_list)
        seg_recall_std5fold = np.std(seg_recall_list)


        seg_dice_average5fold=sum(seg_dice_list)/len(seg_dice_list)
        seg_dice_std5fold = np.std(seg_dice_list)

        seg_acc_average5fold=sum(seg_acc_list)/len(seg_acc_list)
        seg_acc_std5fold = np.std(seg_acc_list)

        hd95_average = sum(hd95_list)/len(hd95_list)
        hd95_std = np.std(hd95_list)
                    
        test_seg_info_list.append(['all',seg_iou_average5fold,seg_iou_std5fold,
                                    seg_recall_average5fold,seg_recall_std5fold,
                                    seg_dice_average5fold,seg_dice_std5fold,
                                    seg_acc_average5fold,seg_acc_std5fold,
                                    hd95_average,hd95_std,
                                    0,0,
                                    0,0,
                                    0,0])
                                
        xlsl_path = os.path.join(test_result_dir, model_name)
        if not os.path.exists(xlsl_path):
            os.makedirs(xlsl_path)
        save_val_info_to_excel_add_acc_hd95(test_seg_info_list,os.path.join(xlsl_path,'test_seg_info.xlsx'))

