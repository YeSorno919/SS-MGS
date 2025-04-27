import random
from os.path import join
from os import listdir
from torch.utils import data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])
class Dataset_seg_test(data.Dataset):
    def __init__(self,datapath,img_namelist, num_classes,with_fix=False,sbnt=False):
        super(Dataset_seg_test, self).__init__()

        self.with_fix=with_fix
        self.labeled_filenames = []
        if self.with_fix:
            for sub in img_namelist:

                a = sub["Atlas"][0]
                self.labeled_filenames.append(a)

                for i in sub["OtherImg"]:
                    self.labeled_filenames.append(i)
        else:
            if(sbnt==False):
                for img in img_namelist:
                    for i in img:
                        self.labeled_filenames.append(i)
            else:
                for img in img_namelist:
                    self.labeled_filenames.append(img)
        self.num_classes = num_classes

        self.labeled_file_dir = datapath

    def __getitem__(self, index):

        labed_img_name = self.labeled_filenames[index].split(".")[0]
        labed_img = np.load(join(self.labeled_file_dir, 'img_crop_npy', labed_img_name+".npy"))
        labed_img = labed_img / 255.
        labed_img = labed_img.astype(np.float32)
        labed_img = labed_img[np.newaxis, :, :]
        if(self.num_classes == 2):
             labed_lab = np.load(join(self.labeled_file_dir, 'seg_npy', labed_img_name+".npy"))

        if(labed_lab.ndim == 2): 
            labed_lab = labed_lab

        else: 
            labed_lab = labed_lab[:,:,0]


        if(self.num_classes ==2):
            labed_lab = np.where(labed_lab<=0,0,labed_lab)
            labed_lab = np.where(labed_lab>0,1,labed_lab)

        labed_lab = self.to_categorical(labed_lab, self.num_classes)
        labed_lab = labed_lab.astype(np.float32)

        return labed_img, labed_lab, self.labeled_filenames[index]

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
    
class Dataset_train_seg_25percent(data.Dataset):
    def __init__(self, datapath,img_name, num_classes,subject_num = -1,with_fix=False): 
        super(Dataset_train_seg_25percent, self).__init__()
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
        self.possible_indices = list(range(len(self.traindata)))

    def __getitem__(self, index):    
        subject = self.traindata[index]
        if self.with_fix :#if datajson has divided which image is atlas
            img = subject["Atlas"]
            un = random.sample(subject["OtherImg"], 2)
            img = img+un
            
            possible_indices = list(range(len(self.traindata)))

            possible_indices.remove(index)

            random_index = random.choice(possible_indices)
            negative_subject = self.traindata[random_index]
            negative_img_name = random.choice(negative_subject['OtherImg']).split(".")[0]
        else:
            img = random.sample(self.traindata[index], 3)

        
        if(self.with_fix):
            negative_img = np.load(join(self.labeled_file_dir, 'img_crop_npy', negative_img_name+".npy"))
            negative_img = negative_img / 255.
            negative_img = negative_img.astype(np.float32)
            negative_img = negative_img[np.newaxis, :, :]
            
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
        
        
        labed_lab = self.to_categorical(labed_lab, self.num_classes) #2*350*740
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
        
        return labed_img, labed_lab, unlabed_img1, unlabed_img2,l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,negative_img

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
        return len(self.traindata)


class Dataset_train_seg_50_75percent(data.Dataset):
    def __init__(self, datapath,img_name, num_classes,subject_num = -1,with_fix=False): 
        super(Dataset_train_seg_50_75percent, self).__init__()
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
        self.possible_indices = list(range(len(self.traindata)))

    def __getitem__(self, index):    
        subject = self.traindata[index]
        if self.with_fix :
            img = random.sample(subject["Atlas"], 1)[0]
            z = subject["Atlas"].copy()
            z.remove(img)
            t =z+subject["OtherImg"]
            un = random.sample(t, 2)
            tmp = []
            tmp.append(img)
            img=tmp+un
            
            possible_indices = list(range(len(self.traindata)))

            possible_indices.remove(index)

            random_index = random.choice(possible_indices)
            negative_subject = self.traindata[random_index]
            negative_img_name = random.choice(negative_subject['OtherImg']).split(".")[0]
        else:
            img = random.sample(self.traindata[index], 3)
        
        if(self.with_fix):
            negative_img = np.load(join(self.labeled_file_dir, 'img_crop_npy', negative_img_name+".npy"))
            negative_img = negative_img / 255.
            negative_img = negative_img.astype(np.float32)
            negative_img = negative_img[np.newaxis, :, :]
            
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
        
        

        
        return labed_img, labed_lab, unlabed_img1, unlabed_img2,l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,negative_img

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
        return len(self.traindata)
class Dataset_train_seg_100percent(data.Dataset):
    def __init__(self, datapath,img_name, num_classes,subject_num = -1,with_fix=False,spatial_sample_from_other_patient=False,simility_json='',topnum = 1): 
        super(Dataset_train_seg_100percent, self).__init__()
        self.spatial_sample_from_other_patient = spatial_sample_from_other_patient
        self.simility_json=simility_json
        self.topnum=topnum
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
        self.possible_indices = list(range(len(self.traindata)))
    def __getitem__(self, index):    
        subject = self.traindata[index]
        if self.with_fix :
            img = subject["Atlas"]
            un = subject["OtherImg"]
            img = img+un
            img = random.sample(img, 3)
            
            possible_indices = list(range(len(self.traindata)))

            possible_indices.remove(index)

            random_index = random.choice(possible_indices)
            negative_subject = self.traindata[random_index]

            negative_subject_img = negative_subject['OtherImg']+negative_subject['Atlas']
            negative_img_name = random.choice(negative_subject_img).split(".")[0]
        else:
            img = random.sample(self.traindata[index], 3)
            
        labed_img_name = img[0].split(".")[0]
        unlabed_img1_name = img[1].split(".")[0] 
        unlabed_img2_name = img[2].split(".")[0] 
        if(self.with_fix):
            negative_img = np.load(join(self.labeled_file_dir, 'img_crop_npy', negative_img_name+".npy"))
            negative_img = negative_img / 255.
            negative_img = negative_img.astype(np.float32)
            negative_img = negative_img[np.newaxis, :, :]

        if(self.spatial_sample_from_other_patient):

            with open(self.simility_json, "r", encoding="utf-8") as fp:
                simility = json.load(fp)

                sim_img_name_list = simility[labed_img_name][:self.topnum]

                unlabed_img2_name_return = random.choice(sim_img_name_list)[0]

                unlabed_img2_name = unlabed_img2_name_return.split(".")[0]

        
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
        l_u1_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', labed_img_name_to_unlabed_img1_name+".npy"))
        l_u1_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', labed_img_name_to_unlabed_img1_name+"_comptx.npy"))
        u1_l_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img1_to_labed_img_name+"_comptx.npy"))
        
        
        if(self.spatial_sample_from_other_patient):
            u2_l_img = np.load(join(self.labeled_file_dir, 'inter_subject_affine_img_npy', unlabed_img2_to_labed_img_name+".npy"))    
            u1_u2_img = np.load(join(self.labeled_file_dir, 'inter_subject_affine_img_npy', unlabed_img1_to_unlabed_img2_name+".npy"))
            u2_u1_img = np.load(join(self.labeled_file_dir, 'inter_subject_affine_img_npy', unlabed_img2_to_unlabed_img1_name+".npy"))
            l_u2_img = np.load(join(self.labeled_file_dir, 'inter_subject_affine_img_npy', labed_img_name_to_unlabed_img2_name+".npy"))
            u2_l_displace_field = np.load(join(self.labeled_file_dir, 'inter_subject_affine_displacement_field', unlabed_img2_to_labed_img_name+"_comptx.npy"))
            u1_u2_displace_field = np.load(join(self.labeled_file_dir, 'inter_subject_affine_displacement_field', unlabed_img1_to_unlabed_img2_name+"_comptx.npy"))
            u2_u1_displace_field = np.load(join(self.labeled_file_dir, 'inter_subject_affine_displacement_field', unlabed_img2_to_unlabed_img1_name+"_comptx.npy"))
            l_u2_displace_field = np.load(join(self.labeled_file_dir, 'inter_subject_affine_displacement_field', labed_img_name_to_unlabed_img2_name+"_comptx.npy"))
        else:
            u2_l_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img2_to_labed_img_name+".npy"))    
            u1_u2_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img1_to_unlabed_img2_name+".npy"))
            u2_u1_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', unlabed_img2_to_unlabed_img1_name+".npy"))
            l_u2_img = np.load(join(self.labeled_file_dir, 'all_pair_affine_img_npy', labed_img_name_to_unlabed_img2_name+".npy"))
            u2_l_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img2_to_labed_img_name+"_comptx.npy"))  
            u1_u2_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img1_to_unlabed_img2_name+"_comptx.npy"))
            u2_u1_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', unlabed_img2_to_unlabed_img1_name+"_comptx.npy"))
            l_u2_displace_field = np.load(join(self.labeled_file_dir, 'all_pair_affine_displacement_field', labed_img_name_to_unlabed_img2_name+"_comptx.npy"))
        
        
        
        if(self.spatial_sample_from_other_patient):
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
        
        
        labed_lab = self.to_categorical(labed_lab, self.num_classes) #2*350*740
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

        
        return labed_img, labed_lab, unlabed_img1, unlabed_img2,l_u1_img,l_u2_img,u1_l_img,u2_l_img,u1_u2_img,u2_u1_img,l_u1_seg,l_u2_seg,l_u1_displace_field,l_u2_displace_field,u1_l_displace_field,u2_l_displace_field,u1_u2_displace_field,u2_u1_displace_field,negative_img,img[0],img[1],unlabed_img2_name_return

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
        return len(self.traindata)


