# ⭐Semi-supervised Infrared Meibomian Gland Segmentation with Intra-patient Registration and Feature Supervision

This is the code for the paper "Semi-supervised Infrared Meibomian Gland Segmentation with Intra-patient Registration and Feature Supervision", which has been submitted to ICIP 2025.

## 💡 Introduction
 



**🖼️The Overall Structural Diagram**

![Fig1](https://github.com/user-attachments/assets/c493bae3-1730-4593-a18e-6d43c11920c5)

**📝Abstract**

  Low-cost and high-precision infrared meibomian gland segmentation is an important basis for early diagnosis and monitoring of many ocular diseases in ophthalmic clinical practice. To address the issue of limited labeled data, we propose a novel semi-supervised meibomian gland segmentation approach. By leveraging the prior knowledge of the patient each image belongs to, intra-patient registration is taken to generate diverse and lifelike pseudo-labeled data. Contrastive learning strategy with reliable negative sample filtering is also introduced to resolve the insufficient supervision in the feature space. Experimental results on the private dataset demonstrate the success of the proposed approach, exhibiting superiority over the state-of-the-art methods.
  
  🎉 **Our Contributions**

- We developed a registration sampling generation module (RSGM). By conducting registration and sampling on intra-patient data, better diversity and authenticity in pseudo annotation generation is achieved.

- We designed contrastive learning with reliable negative sample filtering (RNSF). By learning more discriminative gland features, the lack of supervision in the feature space and poor class separability is compensated.

- We verified the proposed method on private dataset, showing its performance superiority over mainstream semi-supervised segmentation methods.

  

## 🏕️ Dataset
  The dataset, provided by the Ophthalmology Department of the Fujian Provincial Hospital, includes 292 infrared images of upper eyelid meibomian glands, collected from 73 MGD patients( 4 images per patient) with a resolution of
740 × 350 pixels. All images were pixel-level annotated by ophthalmologists. The dataset was then split into training, validation, and test sets in a 7:1:2 ratio. Each patient’s data was strictly limited to appear in just one set.
  We assume the data folder (`Datasets`) has the following structure: 
```
# Provide the organization structure of the dataset.
such as:
Datasets
├── <MGD> 
│ └── Image
│   └── ...
│ └── Label
│   └── ...
| └──train_dict1.json
| └──train_dict2.json
| └──train_dict3.json
| └──train_dict4.json
| └──train_dict5.json
```

## 🚀 Quick Start 🔥🔥🔥


## 🛠️ Environment

Training and evaluation environment:  Python 3.9.19,  PyTorch 2.3.1,  CUDA 12.1.  Run the following command to install required packages.

```
pip install -r requirements.txt
```



## 🔨Training 
### Train Registration Model
The `5fold_Train_Reg.py` is used to train the registration model for subsequent training segmentation.
Through `5fold_Test_Reg_or_Seg.py`, test the registration model trained above, and find the best model weights in the set output path.
### Train Segmentation Model
Use `5fold_Train_SS_MGS.py` to train the segmentation model, and pay attention to the address that gives the weight of the registration model.
Through `5fold_Test_Reg_or_Seg.py`, the segmentation model obtained by testing the above training

 ⚠️ Note: Because the registration model weights are used when training segmentation, you need to train the registration model first.



## 🏗️ Evaluation
Run the `5fold_Test_Reg_or_Seg.py`, test the registration model trained above, and find the best model weights in the set output path. Calculate IoU, Dice to evaluate the performance of registration model. 

Run the `5fold_Test_Reg_or_Seg.py`, test the segmentation model trained above. Calculate IoU, Dice, HD95 to evaluate the performance of segmentation model. 



## 🌟 Acknowledgements

- This repository is built upon [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [VoxelMorph](https://github.com/voxelmorph/voxelmorph), [BRBS](https://github.com/YutingHe-list/BRBS), [ANTsPy](https://github.com/ANTsX/ANTsPy). Thank the authors of these open source repositories for their efforts. And thank the ACs and reviewers for their effort when dealing with our paper.
## 🖊️ Citation
If you find this repository helpful, please consider citing our paper.

```
None
```

