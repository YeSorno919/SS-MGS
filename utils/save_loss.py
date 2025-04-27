import os

import pandas as pd

def save_training_info_to_excel(sim_loss_number,train_info_list, file_path):
    if(sim_loss_number==1):
        data = {
            'Epoch': [],
            'Iter': [],
            'L_smooth': [],
            'L_sim': [],
            'L_SeC': [],
            'L_i': [],
            'L_mix': [],
            'L_seg': []
        }

        for train_info in train_info_list:
            epoch, iter, L_smooth, L_sim, L_SeC, L_i, L_mix, L_seg = train_info
            data['Epoch'].append(epoch)
            data['Iter'].append(iter)
            data['L_smooth'].append(L_smooth)
            data['L_sim'].append(L_sim)
            data['L_SeC'].append(L_SeC)
            data['L_i'].append(L_i)
            data['L_mix'].append(L_mix)
            data['L_seg'].append(L_seg)
    elif(sim_loss_number==2):
        data = {
            'Epoch': [],
            'Iter': [],
            'L_smooth': [],
            'L_sim': [],
            'L_sim_2': [],
            'L_SeC': [],
            'L_i': [],
            'L_mix': [],
            'L_seg': []
        }
        for train_info in train_info_list:
            epoch, iter, L_smooth, L_sim,L_sim_2, L_SeC, L_i, L_mix, L_seg = train_info
            data['Epoch'].append(epoch)
            data['Iter'].append(iter)
            data['L_smooth'].append(L_smooth)
            data['L_sim'].append(L_sim)
            data['L_sim_2'].append(L_sim_2)
            data['L_SeC'].append(L_SeC)
            data['L_i'].append(L_i)
            data['L_mix'].append(L_mix)
            data['L_seg'].append(L_seg)

    df = pd.DataFrame(data)
    if not os.path.exists(file_path):
        open(file_path, 'w').close()

    df.to_excel(file_path, index=False)



def save_val_info_to_excel(val_info_list, file_path):

    data = {
        'Epoch': [],
        'Seg_IoU_average': [],
        'Seg_IoU_error': [],
        'Seg_Recall_average': [],
        'Seg_Recall_error': [],
        'Seg_Dice_average': [],
        'Seg_Dice_error': [],
        'Reg_IoU_average': [],
        'Reg_IoU_error': [],
        'Reg_Recall_average': [],
        'Reg_Recall_error': [],
        'Reg_Dice_average': [],
        'Reg_Dice_error': [],

    }

    for val_info in val_info_list:
        epoch, Seg_IoU_average, Seg_IoU_error, Seg_Recall_average, Seg_Recall_error, Seg_Dice_average,Seg_Dice_error, Reg_IoU_average, Reg_IoU_error, Reg_Recall_average,Reg_Recall_error,Reg_Dice_average,Reg_Dice_error= val_info
        data['Epoch'].append(epoch)
        data['Seg_IoU_average'].append(Seg_IoU_average)
        data['Seg_IoU_error'].append(Seg_IoU_error)
        data['Seg_Recall_average'].append(Seg_Recall_average)
        data['Seg_Recall_error'].append(Seg_Recall_error)
        data['Seg_Dice_average'].append(Seg_Dice_average)
        data['Seg_Dice_error'].append(Seg_Dice_error)
        data['Reg_IoU_average'].append(Reg_IoU_average)
        data['Reg_IoU_error'].append(Reg_IoU_error)
        data['Reg_Recall_average'].append(Reg_Recall_average)
        data['Reg_Recall_error'].append(Reg_Recall_error)
        data['Reg_Dice_average'].append(Reg_Dice_average)
        data['Reg_Dice_error'].append(Reg_Dice_error)


    df = pd.DataFrame(data)
    if not os.path.exists(file_path):
        open(file_path, 'w').close()

    df.to_excel(file_path, index=False,float_format='%.4f')
    
    
def save_val_info_to_excel_add_acc(val_info_list, file_path):

    data = {
        'Epoch': [],
        'Seg_IoU_average': [],
        'Seg_IoU_error': [],
        'Seg_Recall_average': [],
        'Seg_Recall_error': [],
        'Seg_Dice_average': [],
        'Seg_Dice_error': [],
        'Seg_Acc_average': [],
        'Seg_Acc_error': [],
        'Reg_IoU_average': [],
        'Reg_IoU_error': [],
        'Reg_Recall_average': [],
        'Reg_Recall_error': [],
        'Reg_Dice_average': [],
        'Reg_Dice_error': [],

    }

    for val_info in val_info_list:
        epoch, Seg_IoU_average, Seg_IoU_error, Seg_Recall_average, Seg_Recall_error, Seg_Dice_average,Seg_Dice_error, Seg_Acc_average,Seg_Acc_error,Reg_IoU_average, Reg_IoU_error, Reg_Recall_average,Reg_Recall_error,Reg_Dice_average,Reg_Dice_error= val_info
        data['Epoch'].append(epoch)
        data['Seg_IoU_average'].append(Seg_IoU_average)
        data['Seg_IoU_error'].append(Seg_IoU_error)
        data['Seg_Recall_average'].append(Seg_Recall_average)
        data['Seg_Recall_error'].append(Seg_Recall_error)
        data['Seg_Dice_average'].append(Seg_Dice_average)
        data['Seg_Dice_error'].append(Seg_Dice_error)
        data['Seg_Acc_average'].append(Seg_Acc_average)
        data['Seg_Acc_error'].append(Seg_Acc_error)
        data['Reg_IoU_average'].append(Reg_IoU_average)
        data['Reg_IoU_error'].append(Reg_IoU_error)
        data['Reg_Recall_average'].append(Reg_Recall_average)
        data['Reg_Recall_error'].append(Reg_Recall_error)
        data['Reg_Dice_average'].append(Reg_Dice_average)
        data['Reg_Dice_error'].append(Reg_Dice_error)

    df = pd.DataFrame(data)
    if not os.path.exists(file_path):
        open(file_path, 'w').close()
    df.to_excel(file_path, index=False,float_format='%.4f')
    
    
def save_val_info_to_excel_add_acc_hd95(val_info_list, file_path):

    data = {
        'Epoch': [],
        'Seg_IoU_average': [],
        'Seg_IoU_error': [],
        'Seg_Recall_average': [],
        'Seg_Recall_error': [],
        'Seg_Dice_average': [],
        'Seg_Dice_error': [],
        'Seg_Acc_average': [],
        'Seg_Acc_error': [],
        'HD95_average': [],
        'HD95_error': [],
        'Reg_IoU_average': [],
        'Reg_IoU_error': [],
        'Reg_Recall_average': [],
        'Reg_Recall_error': [],
        'Reg_Dice_average': [],
        'Reg_Dice_error': [],

    }

    for val_info in val_info_list:
        epoch, Seg_IoU_average, Seg_IoU_error, Seg_Recall_average, Seg_Recall_error, Seg_Dice_average,Seg_Dice_error, Seg_Acc_average,Seg_Acc_error,HD95_average,HD95_error,Reg_IoU_average, Reg_IoU_error, Reg_Recall_average,Reg_Recall_error,Reg_Dice_average,Reg_Dice_error= val_info
        data['Epoch'].append(epoch)
        data['Seg_IoU_average'].append(Seg_IoU_average)
        data['Seg_IoU_error'].append(Seg_IoU_error)
        data['Seg_Recall_average'].append(Seg_Recall_average)
        data['Seg_Recall_error'].append(Seg_Recall_error)
        data['Seg_Dice_average'].append(Seg_Dice_average)
        data['Seg_Dice_error'].append(Seg_Dice_error)
        data['Seg_Acc_average'].append(Seg_Acc_average)
        data['Seg_Acc_error'].append(Seg_Acc_error)
        data['HD95_average'].append(HD95_average)
        data['HD95_error'].append(HD95_error)
        data['Reg_IoU_average'].append(Reg_IoU_average)
        data['Reg_IoU_error'].append(Reg_IoU_error)
        data['Reg_Recall_average'].append(Reg_Recall_average)
        data['Reg_Recall_error'].append(Reg_Recall_error)
        data['Reg_Dice_average'].append(Reg_Dice_average)
        data['Reg_Dice_error'].append(Reg_Dice_error)

    df = pd.DataFrame(data)
    if not os.path.exists(file_path):
        open(file_path, 'w').close()
    df.to_excel(file_path, index=False,float_format='%.4f')