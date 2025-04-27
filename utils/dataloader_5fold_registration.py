import random
from os.path import join
from torch.utils import data
import numpy as np
import itertools
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])
class Dataset_train_reg(data.Dataset):
    def __init__(self, datapath,img_name, num_classes,subject_num = -1,with_fix=False): 
        super(Dataset_train_reg, self).__init__()

        self.num_classes = num_classes
        self.traindata = img_name
        self.with_fix = with_fix
        self.unlabeled_file_dir = datapath
        self.labeled_file_dir = datapath
        if subject_num!=-1:
            self.subject_num = min(subject_num,len(self.traindata))
        else:
            self.subject_num = len(self.traindata) 
            
        self.traindata  = img_name[:self.subject_num]

    def __getitem__(self, index):
        
        random_index = np.random.randint(low=0, high=len(self.traindata))
        
        subject = self.traindata[random_index]
        if self.with_fix :
            img = subject["Atlas"]
            un = random.sample(subject["OtherImg"], 2)
            img = img+un

        else:
            img = random.sample(self.traindata[random_index], 3)

        
        labed_img_name = img[0].split(".")[0]
        unlabed_img1_name = img[1].split(".")[0]
        unlabed_img2_name = img[2].split(".")[0]
        
        labed_img = np.load(join(self.labeled_file_dir, 'img_crop_npy', labed_img_name+".npy"))
        unlabed_img1 = np.load(join(self.labeled_file_dir, 'img_crop_npy', unlabed_img1_name+".npy"))
        unlabed_img2 = np.load(join(self.labeled_file_dir, 'img_crop_npy', unlabed_img2_name+".npy"))
        
        unlabed_img1_to_labed_img_name = unlabed_img1_name+"to"+labed_img_name
        unlabed_img2_to_labed_img_name = unlabed_img2_name+"to"+labed_img_name
        labed_img_name_to_unlabed_img1_name = labed_img_name+"to"+unlabed_img1_name
        labed_img_name_to_unlabed_img2_name = labed_img_name+"to"+unlabed_img2_name
        unlabed_img1_to_unlabed_img2_name = unlabed_img1_name+"to"+unlabed_img2_name
        unlabed_img2_to_unlabed_img1_name = unlabed_img2_name+"to"+unlabed_img1_name
        
        u1_l_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img1_to_labed_img_name+".npy"))
        u2_l_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img2_to_labed_img_name+".npy"))
        u1_u2_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img1_to_unlabed_img2_name+".npy"))
        u2_u1_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img2_to_unlabed_img1_name+".npy"))
        l_u1_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', labed_img_name_to_unlabed_img1_name+".npy"))
        l_u2_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', labed_img_name_to_unlabed_img2_name+".npy"))
        
        u1_l_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img1_to_labed_img_name+"_comptx.npy"))
        u2_l_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img2_to_labed_img_name+"_comptx.npy"))
        u1_u2_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img1_to_unlabed_img2_name+"_comptx.npy"))
        u2_u1_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img2_to_unlabed_img1_name+"_comptx.npy"))
        l_u1_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', labed_img_name_to_unlabed_img1_name+"_comptx.npy"))
        l_u2_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', labed_img_name_to_unlabed_img2_name+"_comptx.npy"))
        

        
        if(self.num_classes == 2):
            labed_lab = np.load(join(self.labeled_file_dir, 'seg_npy', labed_img_name+".npy"))
            l_u1_seg=np.load(join(self.labeled_file_dir, 'all_pair_affine_seg_npy', labed_img_name_to_unlabed_img1_name+".npy"))
            l_u2_seg=np.load(join(self.labeled_file_dir, 'all_pair_affine_seg_npy', labed_img_name_to_unlabed_img2_name+".npy"))
        if(self.num_classes == 2):
            labed_lab = np.where(labed_lab<=0,0,labed_lab)
            labed_lab = np.where(labed_lab>0,1,labed_lab)
            
            l_u1_seg = np.where(l_u1_seg<=0,0,l_u1_seg)
            l_u1_seg = np.where(l_u1_seg>0,1,l_u1_seg)
            
            l_u2_seg = np.where(l_u2_seg<=0,0,l_u2_seg)
            l_u2_seg = np.where(l_u2_seg>0,1,l_u2_seg)

        labed_img = labed_img / 255.
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :]
        
        
        labed_lab = self.to_categorical(labed_lab, self.num_classes)
        labed_lab = labed_lab.astype(np.float32)
        
        l_u1_seg = self.to_categorical(l_u1_seg, self.num_classes)
        l_u1_seg = l_u1_seg.astype(np.float32)
        
        l_u2_seg = self.to_categorical(l_u2_seg, self.num_classes)
        l_u2_seg = l_u2_seg.astype(np.float32)

        unlabed_img1 = unlabed_img1 / 255.
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :]

        unlabed_img2 = unlabed_img2 / 255.
        unlabed_img2 = unlabed_img2.astype(np.float32)
        unlabed_img2 = unlabed_img2[np.newaxis, :, :]
        
        u1_l_img = u1_l_img/255
        u1_l_img = u1_l_img.astype(np.float32)
        u1_l_img = u1_l_img[np.newaxis, :, :]
        
        u2_l_img = u2_l_img/255
        u2_l_img = u2_l_img.astype(np.float32)
        u2_l_img = u2_l_img[np.newaxis, :, :]
        
        u1_u2_img = u1_u2_img/255
        u1_u2_img = u1_u2_img.astype(np.float32)
        u1_u2_img = u1_u2_img[np.newaxis, :, :]
        
        u2_u1_img = u2_u1_img/255
        u2_u1_img = u2_u1_img.astype(np.float32)
        u2_u1_img = u2_u1_img[np.newaxis, :, :]
        
        l_u1_img = l_u1_img/255
        l_u1_img = l_u1_img.astype(np.float32)
        l_u1_img = l_u1_img[np.newaxis, :, :]
        
        l_u2_img = l_u2_img/255
        l_u2_img = l_u2_img.astype(np.float32)
        l_u2_img = l_u2_img[np.newaxis, :, :]

        return labed_img, labed_lab, unlabed_img1, unlabed_img2,l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field


    def to_categorical(self, y, num_classes=None):
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

    def __len__(self):
        length = 0
        for i in self.traindata:
            length += len(i)
        return length
class Dataset_reg_test(data.Dataset):
    def __init__(self, datapath,img_namelist, num_classes,with_fix = False):
        super(Dataset_reg_test, self).__init__()
        self.num_classes = num_classes
        self.testdata = img_namelist
        self.with_fix = with_fix
        self.unlabeled_file_dir = datapath
        self.labeled_file_dir = datapath
        labeled_filenames = [] 

        for data in self.testdata:
            if self.with_fix:
                combinations = [(a, b) for a in data["OtherImg"] for b in data["Altas"]]
            else: combinations = list(itertools.combinations(data, 2))
            
            for i in combinations:
                labeled_filenames.append(i)
        self.labeled_filenames = labeled_filenames

    def __getitem__(self, index):
        label_img1_name = self.labeled_filenames[index][0].split('.')[0]
        labed_img1 = np.load(join(self.labeled_file_dir, 'img_crop_npy',label_img1_name+'.npy' ))
        labed_img1 = labed_img1 / 255.
        labed_img1 = labed_img1.astype(np.float32)
        labed_img1 = labed_img1[np.newaxis, :, :]

        if(self.num_classes ==2):
            labed_lab1 = np.load(join(self.labeled_file_dir, 'seg_npy', label_img1_name+'.npy'))

        if(labed_lab1.ndim == 2): 
            labed_lab1 = labed_lab1
        else: 
            labed_lab1 = labed_lab1[:,:,0]

        if(self.num_classes ==2):
            labed_lab1 = np.where(labed_lab1 <=0, 0, labed_lab1)
            labed_lab1 = np.where(labed_lab1 >0, 1, labed_lab1)

        labed_lab1 = self.to_categorical(labed_lab1, self.num_classes)
        labed_lab1 = labed_lab1.astype(np.float32)
        
        label_img2_name = self.labeled_filenames[index][1].split('.')[0]
        labed_img2 = np.load(join(self.labeled_file_dir, 'img_crop_npy',label_img2_name+'.npy' ))
        
        labed_img2 = labed_img2 / 255.
        labed_img2 = labed_img2.astype(np.float32)
        labed_img2 = labed_img2[np.newaxis, :, :]

        if(self.num_classes == 2):
            labed_lab2 = np.load(join(self.labeled_file_dir, 'seg_npy', label_img2_name+'.npy'))

        if(labed_lab2.ndim == 2): 
            labed_lab2 = labed_lab2
        else: 
            labed_lab2 = labed_lab2[:,:,0]
        if(self.num_classes == 2):
            labed_lab2 = np.where(labed_lab2<=0,0,labed_lab2)
            labed_lab2 = np.where(labed_lab2>0,1,labed_lab2)


        labed_lab2 = self.to_categorical(labed_lab2, self.num_classes)
        labed_lab2 = labed_lab2.astype(np.float32)

        labed_img1_to_labed_img2_name = label_img1_name+"to"+label_img2_name
        labed_img2_to_labed_img1_name = label_img2_name+"to"+label_img1_name
        
        l1_l2_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', labed_img1_to_labed_img2_name+".npy"))
        l2_l1_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', labed_img2_to_labed_img1_name+".npy"))
        
        l1_l2_img = l1_l2_img/255
        l1_l2_img = l1_l2_img.astype(np.float32)
        l1_l2_img = l1_l2_img[np.newaxis, :, :]
        
        l2_l1_img = l2_l1_img/255
        l2_l1_img = l2_l1_img.astype(np.float32)
        l2_l1_img = l2_l1_img[np.newaxis, :, :]

        l1_l2_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', labed_img1_to_labed_img2_name+"_comptx.npy"))
        l2_l1_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', labed_img2_to_labed_img1_name+"_comptx.npy"))
        if self.num_classes==2:

            l1_l2_seg = np.load(join(self.labeled_file_dir, 'all_pair_affine_seg_npy', labed_img1_to_labed_img2_name+".npy"))
            l2_l1_seg = np.load(join(self.labeled_file_dir, 'all_pair_affine_seg_npy', labed_img2_to_labed_img1_name+".npy"))
            
            l1_l2_seg = np.where(l1_l2_seg<=0,0,l1_l2_seg)
            l1_l2_seg = np.where(l1_l2_seg>0,1,l1_l2_seg)
            
            l2_l1_seg = np.where(l2_l1_seg<=0,0,l2_l1_seg)
            l2_l1_seg = np.where(l2_l1_seg>0,1,l2_l1_seg)
            
        l1_l2_seg = self.to_categorical(l1_l2_seg, self.num_classes)   
        l1_l2_seg = l1_l2_seg.astype(np.float32) 
    
        l2_l1_seg = self.to_categorical(l2_l1_seg, self.num_classes)   
        l2_l1_seg = l2_l1_seg.astype(np.float32)
 
        return labed_img1, labed_lab1, labed_img2, labed_lab2, \
                l1_l2_img,l2_l1_img,l1_l2_seg,l2_l1_seg,\
                l1_l2_displace_field,l2_l1_displace_field,\
               label_img1_name, label_img2_name

    def to_categorical(self, y, num_classes=None):
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

    def __len__(self):

        return len(self.labeled_filenames)