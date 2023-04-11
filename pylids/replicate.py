from typing_extensions import assert_never
from . import utils
from . import augment
import os
import cv2
import glob
import pandas as pd
from numpy.polynomial import polynomial as P
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile


############################################
### ---  Aug conversion to DLC kpts  --- ###
############################################


def get_augmented_kpts(selected_aug, trn_labeled_data):
    aug, sub_num, file = selected_aug.split('/')[-1].split('_')
    if 's' in sub_num:
        if sub_num == 's1':
            csv_file = glob.glob(os.path.join(trn_labeled_data,sub_num+'_6')+'/*.csv')
        elif sub_num == 's6':
            csv_file = glob.glob(os.path.join(trn_labeled_data,sub_num+'_12')+'/*.csv')
    else:        
        csv_file = glob.glob(os.path.join(trn_labeled_data,sub_num+'_1_1')+'/*.csv')

    csv_df = pd.read_csv(csv_file[0],skiprows=2)
    file_names = csv_df['coords']
    data = csv_df.to_numpy()
    data = data[:,1:]

    for i, file_name in enumerate(file_names):
        if file == file_name.split('/')[-1]:
            keypoints = data[i,:]
    if aug[:-1]=='rot':
        frame = cv2.imread(selected_aug)
        x_kpts, y_kpts = keypoints[::2], keypoints[1::2]
        keypoints = augment.conv_to_iaug_kpts(x_labels=x_kpts,y_labels=y_kpts, frame=frame)
        frame, keypoints = augment.add_rotation(frame,key_pts = keypoints, it=int(aug[-1]))
        keypoints = augment.conv_to_dlc_kpts(keypoints)

    return keypoints


def augmentations_to_dlc_kpts(selected_augs_list,
                              trn_labeled_data,
                              only_eyelids = True,
                              copy_images = False,
                              data_folder= None,
                              save_folder='./',
                              csv_filename='CollectedData_POMlab.csv'):                  
    """
    Args:
    Returns:
    """
    # total DLC keypoints
    if only_eyelids:
        total_keypoints = 64
    else:
        total_keypoints = 96

    # rows 1,2 and 3 of the data frame have idiosyncratic headers as required by DLC
    row1 = np.hstack(('scorer', np.full(total_keypoints, 'POMlab')))
    if only_eyelids:
        row2 = ['bodyparts', 'Lcorner', 'Lcorner', 'Rcorner', 'Rcorner', 'upper1', 'upper1', 'upper2', 'upper2', 'upper3', 'upper3', 'upper4', 'upper4', 'upper5', 'upper5', 'upper6', 'upper6', 'upper7', 'upper7', 'upper8', 'upper8', 'upper9', 'upper9', 'upper10', 'upper10', 'upper11', 'upper11', 'upper12', 'upper12', 'upper13', 'upper13', 'upper14',
                'upper14', 'upper15', 'upper15', 'lower1', 'lower1', 'lower2', 'lower2', 'lower3', 'lower3', 'lower4', 'lower4', 'lower5', 'lower5', 'lower6', 'lower6', 'lower7', 'lower7', 'lower8', 'lower8', 'lower9', 'lower9', 'lower10', 'lower10', 'lower11', 'lower11', 'lower12', 'lower12', 'lower13', 'lower13', 'lower14', 'lower14', 'lower15', 'lower15']
    else:
         row2 = ['bodyparts', 'Lcorner', 'Lcorner', 'Rcorner', 'Rcorner', 'upper1', 'upper1', 'upper2', 'upper2', 'upper3', 'upper3', 'upper4', 'upper4', 'upper5', 'upper5', 'upper6', 'upper6', 'upper7', 'upper7', 'upper8', 'upper8', 'upper9', 'upper9', 'upper10', 'upper10', 'upper11', 'upper11', 'upper12', 'upper12', 'upper13', 'upper13', 'upper14',
                 'upper14', 'upper15', 'upper15', 'lower1', 'lower1', 'lower2', 'lower2', 'lower3', 'lower3', 'lower4', 'lower4', 'lower5', 'lower5', 'lower6', 'lower6', 'lower7', 'lower7', 'lower8', 'lower8', 'lower9', 'lower9', 'lower10', 'lower10', 'lower11', 'lower11', 'lower12', 'lower12', 'lower13', 'lower13', 'lower14', 'lower14', 'lower15', 'lower15',
                 'pLeft', 'pLeft', 'pRight', 'pRight', 'pU1', 'pU1', 'pU2', 'pU2', 'pU3', 'pU3', 'pU4', 'pU4', 'pU5', 'pU5', 'pU6', 'pU6', 'pU7', 'pU7', 'pL1', 'pL1', 'pL2', 'pL2', 'pL3', 'pL3', 'pL4', 'pL4', 'pL5', 'pL5', 'pL6', 'pL6', 'pL7' ,'pL7']
    
    temp = []
    for i in range(total_keypoints):
        if i % 2 == 0:
            temp = np.append(temp, 'x')
        else:
            temp = np.append(temp, 'y')
    row3 = np.hstack(('coords', temp))
    flag = False

    #check if data folder exists before writing
    if os.path.isdir(os.path.join(save_folder,data_folder)) == False: 
        os.makedirs(os.path.join(save_folder,data_folder))

    # Main loop
    for selected_aug in selected_augs_list:
        keypoints = get_augmented_kpts(selected_aug,trn_labeled_data)
        
        if data_folder is not None:
            file_name = 'labeled-data/'+ data_folder+'/'+ selected_aug.split('/')[-1]
        else:
            file_name = selected_aug.split('/')[-1]

        # appends the filename before the 64 keypoint resulting in 65 columns
        df_row = np.hstack((file_name, keypoints))

        if flag == False:
            df = df_row
            flag = True
        else:
            df = np.vstack((df, df_row))
        if copy_images:
            #Copying image frames into DLC folder/labeled-data folder
            copyfile(selected_aug, os.path.join(save_folder,data_folder,selected_aug.split('/')[-1]))

    # stacks rows to make the required dataframe adding the 3 headers before the
    df = np.vstack((row1, row2, row3, df))
    pd_df = pd.DataFrame(df)

    if os.path.isfile(os.path.join(save_folder,data_folder,csv_filename)):
        resp = input('File exists, do you want to overwrite? (y/n) \n')
    if os.path.isfile(os.path.join(save_folder,data_folder,csv_filename)) == False or resp == 'y':
        pd.DataFrame.to_csv(pd_df, os.path.join(save_folder,data_folder,csv_filename),
                            header=False, index=False)
    return pd_df    

############################################
### ---  VEDB conversion to DLC kpts  --- ###
############################################


def get_eyelid_pupil_outlines(frame):
    a = np.where(frame==0,frame,10)
    b = np.where(frame==3,10,0)
    img = np.zeros((400,400,2))
    eyelid_pupil = [] #checks for points common along the eyelid and pupil
    for i in range(400):
        outline_eye = np.argwhere(a[:,i]==10)
        if len(outline_eye) != 0:
            up_eye = np.max(np.argwhere(a[:,i]==10))
            lo_eye = np.min(np.argwhere(a[:,i]==10))
            img[up_eye,i,0]=50
            img[lo_eye,i,0]=75

        outline_pupil = np.argwhere(b[:,i]==10)
        if len(outline_pupil) != 0:
            up_pupil = np.max(np.argwhere(b[:,i]==10))
            lo_pupil = np.min(np.argwhere(b[:,i]==10))
            img[up_pupil,i,1]=100
            img[lo_pupil,i,1]=100       
            if up_eye == up_pupil:
                eyelid_pupil.append([up_pupil,i])
            if lo_eye == lo_pupil:
                eyelid_pupil.append([lo_pupil,i])
                
        outline_pupil = np.argwhere(b[i,:]==10)
        if len(outline_pupil) != 0:
            up_pupil = np.max(np.argwhere(b[i,:]==10))
            lo_pupil = np.min(np.argwhere(b[i,:]==10))
            img[i,up_pupil,1]=100
            img[i,lo_pupil,1]=100
            if img[i,up_pupil,0]== 50 or img[i,up_pupil,0]== 75:
                eyelid_pupil.append([i,up_pupil])
            if img[i,lo_pupil,0]== 50 or img[i,lo_pupil,0]== 75:
                eyelid_pupil.append([i,lo_pupil])

    return img, eyelid_pupil 
    
    
def split_pupil(img):
    img = img[:,:,1]
    pupil_idx_y = np.argwhere(img==100)[:,0]
    pupil_max_y = np.max(pupil_idx_y)
    pupil_min_y = np.min(pupil_idx_y)
    pupil_mid = (pupil_max_y - pupil_min_y )/2
    pupil_up_idx =np.argwhere(pupil_idx_y<(pupil_max_y-pupil_mid))
    pupil_lo_idx =np.argwhere(pupil_idx_y>(pupil_max_y-pupil_mid))
    pupil_idx = np.argwhere(img==100)

    for i in range(len(pupil_lo_idx)):
        im_x, im_y = pupil_idx[np.squeeze(pupil_lo_idx),:][i]
        img[im_x,im_y] = 100

    for i in range(len(pupil_up_idx)):
        im_x, im_y = pupil_idx[np.squeeze(pupil_up_idx),:][i]
        img[im_x, im_y] = 125

    return img


def generate_vedb_kpts(selected_frame,path_to_seg_maps):
    selected_label = selected_frame.split('/')[-1].split('.')[0]+'_array.npy' 
    frame_label = np.load(os.path.join(path_to_seg_maps,selected_label))

    if selected_label.split('.')[0].split('_')[-2] == 'L':
        frame_label = np.flip(frame_label)
    img, eyelid_pupil = get_eyelid_pupil_outlines(frame_label)
    img_pupil = split_pupil(img)
    img_eyelids = img[:,:,0]

    eyelid_pupil = np.flip(np.asarray(eyelid_pupil))
    slice_idx = np.ceil(len(np.argwhere(img_pupil==100)[:,1])/9)
    arg_idx = np.argwhere(img_pupil.T==100)
    lower_pupil = arg_idx[0:len(arg_idx):int(slice_idx), :]
    if len(lower_pupil) <9:
        inc=1
        while len(lower_pupil)<9:
            slice_idx = np.ceil(len(np.argwhere(img_pupil==100)[:,1])/(9+inc))
            arg_idx = np.argwhere(img_pupil.T==100)
            lower_pupil = arg_idx[0:len(arg_idx):int(slice_idx), :]
            inc = inc + 1
            
    lower_pupil = lower_pupil[1:,:]

    if len(eyelid_pupil)!= 0:
        new_filt = [0,0]
        for i in range(len(lower_pupil)):
            flag = 0
            for j in range(len(eyelid_pupil)):
                 if all(lower_pupil[i,:] == eyelid_pupil[j,:]):
                    flag = 1
                    new_filt = np.vstack((new_filt, np.array([np.nan,np.nan])))
                    break
            if flag == 0:
                new_filt = np.vstack((new_filt, lower_pupil[i,:]))
        lower_pupil = new_filt[1:,:]

    slice_idx = np.ceil(len(np.argwhere(img_pupil==125)[:,1])/9)
    arg_idx = np.argwhere(img_pupil.T==125)
    arg_idx = np.flip(arg_idx, axis = 0)
    upper_pupil = arg_idx[0:len(arg_idx):int(slice_idx), :]
    
    if len(upper_pupil)<9:
        inc=1
        while len(upper_pupil)<9:
            slice_idx = np.ceil(len(np.argwhere(img_pupil==125)[:,1])/(9+inc))
            arg_idx = np.argwhere(img_pupil.T==125)
            arg_idx = np.flip(arg_idx, axis = 0)
            upper_pupil = arg_idx[0:len(arg_idx):int(slice_idx), :]
            inc=inc+1
    upper_pupil = upper_pupil[1:,:]

    if len(eyelid_pupil)!= 0:        
        new_filt = [0,0]
        for i in range(len(upper_pupil)):
            flag = 0
            for j in range(len(eyelid_pupil)):
                 if all(upper_pupil[i,:] == eyelid_pupil[j,:]):
                    flag = 1
                    new_filt = np.vstack((new_filt, np.array([np.nan,np.nan])))
                    break
            if flag == 0:
                new_filt = np.vstack((new_filt, upper_pupil[i,:]))
        upper_pupil = new_filt[1:,:]
    upper_pupil = np.flip(upper_pupil, axis = 0)

    slice_idx = np.ceil(len(np.argwhere(img_eyelids==50)[:,1])/17)
    arg_idx = np.argwhere(img_eyelids.T==50)
    lower_eyelid = arg_idx[0:len(arg_idx):int(slice_idx), :]
    lower_eyelid = lower_eyelid[:-1,:]
    lower_eyelid = lower_eyelid.astype('float')
    for i, pts_eye in enumerate(lower_eyelid):
       if int(pts_eye[0]) == 399 or int(pts_eye[1]) == 399:
            lower_eyelid[i,0] = np.nan
            lower_eyelid[i,1] = np.nan
       elif int(pts_eye[0]) == 0 or int(pts_eye[1]) == 0:
            lower_eyelid[i,0] = np.nan
            lower_eyelid[i,1] = np.nan 


    slice_idx = np.ceil(len(np.argwhere(img_eyelids==75)[:,1])/17)
    arg_idx = np.argwhere(img_eyelids.T==75)
    arg_idx = np.flip(arg_idx, axis = 0)
    upper_eyelid = arg_idx[0:len(arg_idx):int(slice_idx), :]
    upper_eyelid = upper_eyelid[:-1,:]
    upper_eyelid = upper_eyelid.astype('float')
    for i, pts_eye in enumerate(upper_eyelid):
       if int(pts_eye[0]) == 399 or int(pts_eye[1]) == 399:
            upper_eyelid[i,0] = np.nan
            upper_eyelid[i,1] = np.nan
       
       elif int(pts_eye[0]) == 0 or int(pts_eye[1]) == 0:
            upper_eyelid[i,0] = np.nan
            upper_eyelid[i,1] = np.nan 
        
    upper_eyelid = np.flip(upper_eyelid, axis =0)

    upper_eyelid = np.vstack((lower_eyelid[0,:], upper_eyelid))
    lower_eyelid = lower_eyelid[1:,:]

    upper_pupil = np.vstack((upper_pupil, lower_pupil[-1,:]))
    lower_pupil = lower_pupil[:-1,:]

    idx_upper = [0, 16, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    idx_lower = [7, 3, 11, 1, 5, 9, 13, 0, 2, 4, 6, 8, 10, 12, 14]

    pupil_idx_upper = [0, 8, 4, 2, 6, 1, 3, 5, 7]
    pupil_idx_lower = [3, 1, 5, 0, 2, 4, 6]

    upper_eyelid_rearranged = upper_eyelid[idx_upper]
    lower_eyelid_rearranged = lower_eyelid[idx_lower]

    upper_pupil_rearranged = upper_pupil[pupil_idx_upper]
    lower_pupil_rearranged = lower_pupil[pupil_idx_lower]

    return upper_eyelid_rearranged, lower_eyelid_rearranged, upper_pupil_rearranged, lower_pupil_rearranged


def vedb_to_dlc_kpts(selected_frame_list,
                    path_to_seg_maps,
                    only_eyelids = True,
                    copy_images = False,
                    data_folder=None,
                    save_folder='./',
                    csv_filename='CollectedData_POMlab.csv'):
    # total DLC keypoints
    if only_eyelids:
        total_keypoints = 32 * 2 
    else:
        total_keypoints = 48 * 2

    # rows 1,2 and 3 of the data frame have idiosyncratic headers as required by DLC
    row1 = np.hstack(('scorer', np.full(total_keypoints, 'POMlab')))
    if only_eyelids:
        row2 = ['bodyparts', 'Lcorner', 'Lcorner', 'Rcorner', 'Rcorner', 'upper1', 'upper1', 'upper2', 'upper2', 'upper3', 'upper3', 'upper4', 'upper4', 'upper5', 'upper5', 'upper6', 'upper6', 'upper7', 'upper7', 'upper8', 'upper8', 'upper9', 'upper9', 'upper10', 'upper10', 'upper11', 'upper11', 'upper12', 'upper12', 'upper13', 'upper13', 'upper14',
            'upper14', 'upper15', 'upper15', 'lower1', 'lower1', 'lower2', 'lower2', 'lower3', 'lower3', 'lower4', 'lower4', 'lower5', 'lower5', 'lower6', 'lower6', 'lower7', 'lower7', 'lower8', 'lower8', 'lower9', 'lower9', 'lower10', 'lower10', 'lower11', 'lower11', 'lower12', 'lower12', 'lower13', 'lower13', 'lower14', 'lower14', 'lower15', 'lower15']

    else:
        row2 = ['bodyparts', 'Lcorner', 'Lcorner', 'Rcorner', 'Rcorner', 'upper1', 'upper1', 'upper2', 'upper2', 'upper3', 'upper3', 'upper4', 'upper4', 'upper5', 'upper5', 'upper6', 'upper6', 'upper7', 'upper7', 'upper8', 'upper8', 'upper9', 'upper9', 'upper10', 'upper10', 'upper11', 'upper11', 'upper12', 'upper12', 'upper13', 'upper13', 'upper14',
            'upper14', 'upper15', 'upper15', 'lower1', 'lower1', 'lower2', 'lower2', 'lower3', 'lower3', 'lower4', 'lower4', 'lower5', 'lower5', 'lower6', 'lower6', 'lower7', 'lower7', 'lower8', 'lower8', 'lower9', 'lower9', 'lower10', 'lower10', 'lower11', 'lower11', 'lower12', 'lower12', 'lower13', 'lower13', 'lower14', 'lower14', 'lower15', 'lower15',
            'pLeft', 'pLeft', 'pRight', 'pRight', 'pU1', 'pU1', 'pU2', 'pU2', 'pU3', 'pU3', 'pU4', 'pU4', 'pU5', 'pU5', 'pU6', 'pU6', 'pU7', 'pU7', 'pL1', 'pL1', 'pL2', 'pL2', 'pL3', 'pL3', 'pL4', 'pL4', 'pL5', 'pL5', 'pL6', 'pL6', 'pL7' ,'pL7']
    temp = []
    for i in range(total_keypoints):
        if i % 2 == 0:
            temp = np.append(temp, 'x')
        else:
            temp = np.append(temp, 'y')
    row3 = np.hstack(('coords', temp))
    flag = False
    
    #check if data folder exists before writing
    if os.path.isdir(os.path.join(save_folder,data_folder)) == False: 
        os.makedirs(os.path.join(save_folder,data_folder))

    # Main loop
    for  selected_frame in selected_frame_list:
        upper_eyelid_rearranged, lower_eyelid_rearranged, upper_pupil_rearranged, lower_pupil_rearranged = generate_vedb_kpts(selected_frame,path_to_seg_maps)
    
        # Pairing up x, y coordinates along the columns of each row first for upper then lower eyelids
        for i in range(len(upper_eyelid_rearranged)):
            if i == 0:
                keypoints = [upper_eyelid_rearranged[i,0], upper_eyelid_rearranged[i,1]]
            else:
                keypoints = np.hstack(
                    (keypoints, upper_eyelid_rearranged[i,0], upper_eyelid_rearranged[i,1]))
        for i in range(len(lower_eyelid_rearranged)):
            keypoints = np.hstack(
                (keypoints, lower_eyelid_rearranged[i,0], lower_eyelid_rearranged[i,1]))
                
        if not only_eyelids:
            for i in range(len(upper_pupil_rearranged)):
                keypoints = np.hstack(
                    (keypoints, upper_pupil_rearranged[i,0], upper_pupil_rearranged[i,1]))
            for i in range(len(lower_pupil_rearranged)):
                keypoints = np.hstack(
                    (keypoints, lower_pupil_rearranged[i,0], lower_pupil_rearranged[i,1]))

        if data_folder is not None:
            file_name = 'labeled-data/'+ data_folder+'/'+ selected_frame.split('/')[-1]
        else:
            file_name = selected_frame.split('/')[-1]
        
        # appends the filename before the 64 keypoint resulting in 65 columns
        df_row = np.hstack((file_name, keypoints))

        if flag == False:
            df = df_row
            flag = True
        else:
            df = np.vstack((df, df_row))
        if copy_images:
            if selected_frame.split('.')[0].split('_')[-1] == 'L':
                left_eye_frame = cv2.imread(selected_frame)
                cv2.imwrite(os.path.join(save_folder,data_folder,selected_frame.split('/')[-1]),np.flip(left_eye_frame))
            else:
                #Copying image frames into DLC folder/labeled-data folder
                copyfile(selected_frame, os.path.join(save_folder,data_folder,selected_frame.split('/')[-1]))

    # stacks rows to make the required dataframe adding the 3 headers before the
    df = np.vstack((row1, row2, row3, df))
    pd_df = pd.DataFrame(df)

    if os.path.isfile(os.path.join(save_folder,data_folder,csv_filename)):
        resp = input('File exists, do you want to overwrite? (y/n) \n')
    if os.path.isfile(os.path.join(save_folder,data_folder,csv_filename)) == False or resp == 'y':
        pd.DataFrame.to_csv(pd_df, os.path.join(save_folder,data_folder,csv_filename),
                            header=False, index=False)
    return pd_df

############################################
### ---  Santini conversion to DLC kpts  --- ###
############################################


def load_santini(eyelid_csv='eyelid_data/outlines.csv', eye_image_folder='eyelid_data/data/'):
    """Summary

    Args:
        path_to_csv_file (str, optional): Description
        eye_image_folder (str, optional): Description

    Returns:
        TYPE: Description
    """
    santini_data = np.genfromtxt(eyelid_csv, delimiter=' ')
    sant_files, sant_x, sant_y = santini_data[:,
                                              0], santini_data[:, 1:21:2], santini_data[:, 2:21:2]

    # Flipping the santini data along y axis to match DLC data
    frame = cv2.imread(eye_image_folder + '1.png')
    sant_y = frame.shape[0]-sant_y

    return sant_files, sant_x, sant_y


def resample_rearrange_santini(x, y, files, i, deg=4):
    """Fits a fourth degree polynomial to the santini labels and then resamples 15 linearly sampled keypoints for the upper and lower eyelid
        Also rearranges the resampled fits into the order we expect for retraining DLC aka bisecting keypoints from left to right starting with the corners

    Args:
        x (float): x coordinates for santini labels
        y (float): y coordinates for santini labels
        files (string): filename for give frame
        i (float): frame number to resample
        deg (int, optional): degree of polynomial to fit on eyelids

    Returns:
        float: 15 keypoints for upper and lower eyelids and 2 eyelid corners which are the first and last indices of x y upper resampled vector
    """
    coefs1 = P.polyfit(x[i, :6], y[i, :6], deg, full=False)
    coefs2 = P.polyfit(np.hstack((x[i, 0], x[i, 6:], x[i, 5])), np.hstack(
        (y[i, 0], y[i, 6:], y[i, 5])), deg, full=False)

    x_new = np.linspace(-100, 740, num=1000)
    ffit1 = P.polyval(x_new, coefs1)
    ffit2 = P.polyval(x_new, coefs2)
    idx = np.argwhere(np.diff(np.sign(ffit1 - ffit2))).flatten()
    if len(idx) > 1:  # && within the frame:
        x_new = np.linspace(x_new[int(idx[0])], x_new[int(idx[1])], num=1000)
        ffit1 = P.polyval(x_new, coefs1)
        ffit2 = P.polyval(x_new, coefs2)

    else:
        resample_rearrange_santini(x, y, files, i, deg=deg-1)

    x_resampled_upper = np.linspace(x[i, 0], x[i, 5], 17)
    y_resampled_upper = P.polyval(x_resampled_upper, coefs1)
    x_resampled_lower = np.linspace(
        x_resampled_upper[1], x_resampled_upper[-2], 15)
    y_resampled_lower = P.polyval(x_resampled_lower, coefs2)

    # Rearranging the coordinates according to reflect DLC order
    # Manually added these indices below, I am sure it can be automated but I failed :@
    idx_upper = [0, 16, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    idx_lower = [7, 3, 11, 1, 5, 9, 13, 0, 2, 4, 6, 8, 10, 12, 14]
    x_upper_rearranged = x_resampled_upper[idx_upper]
    y_upper_rearranged = y_resampled_upper[idx_upper]
    x_lower_rearranged = x_resampled_lower[idx_lower]
    y_lower_rearranged = y_resampled_lower[idx_lower]
    png_file = str(int(files[i]))+'.png'

    return x_upper_rearranged, y_upper_rearranged, x_lower_rearranged, y_lower_rearranged, png_file, x_new, ffit1, ffit2


def santini_to_dlc_kpts(selected_frame_list,
                            dataset_csv='santini_dataset/eyelid_data/outlines.csv',
                            eye_images_path='santini_dataset/eyelid_data/data/',
                            copy_images = False,
                            data_folder= None,
                            save_folder='./',
                            csv_filename='CollectedData_POMlab.csv'):
    
    """Wrapper function which takes santini x,y coordinates and file names as input along with the indices of all frames to
       resample, returns a .csv file with resampled keypoints in the format required to fine tune DLC.

    Args:
        sant_files (str): .png file name of files corresponding to given indices
        data_folder (str, optional): location of santini .png wrt to home folder for DLC config file location
        save_folder (str, optional): where to save the .csv files
        csv_filename (str, optional): filename for the saved .csv

    Returns:
        pd_df (panda dataframe): dataframe of the format expected by DLC for fine tuning
    """
    #Loading data from the santini dataset
    sant_files, sant_x, sant_y = load_santini(dataset_csv, eye_images_path)
    
    # total DLC keypoints
    total_keypoints = 64

    # rows 1,2 and 3 of the data frame have idiosyncratic headers as required by DLC
    row1 = np.hstack(('scorer', np.full(total_keypoints, 'POMlab')))
    row2 = ['bodyparts', 'Lcorner', 'Lcorner', 'Rcorner', 'Rcorner', 'upper1', 'upper1', 'upper2', 'upper2', 'upper3', 'upper3', 'upper4', 'upper4', 'upper5', 'upper5', 'upper6', 'upper6', 'upper7', 'upper7', 'upper8', 'upper8', 'upper9', 'upper9', 'upper10', 'upper10', 'upper11', 'upper11', 'upper12', 'upper12', 'upper13', 'upper13', 'upper14',
            'upper14', 'upper15', 'upper15', 'lower1', 'lower1', 'lower2', 'lower2', 'lower3', 'lower3', 'lower4', 'lower4', 'lower5', 'lower5', 'lower6', 'lower6', 'lower7', 'lower7', 'lower8', 'lower8', 'lower9', 'lower9', 'lower10', 'lower10', 'lower11', 'lower11', 'lower12', 'lower12', 'lower13', 'lower13', 'lower14', 'lower14', 'lower15', 'lower15']
    temp = []
    for i in range(total_keypoints):
        if i % 2 == 0:
            temp = np.append(temp, 'x')
        else:
            temp = np.append(temp, 'y')
    row3 = np.hstack(('coords', temp))
    flag = False

    #check if data folder exists before writing
    if os.path.isdir(os.path.join(save_folder,data_folder)) == False: 
        os.makedirs(os.path.join(save_folder,data_folder))

    # Main loop
    for selected_frame in selected_frame_list:
        idx = int(selected_frame.split('/')[-1].split('.')[-2])-1
        # run this function to fit a polynomial and resample and rearrange the santini keypoints
        x_resampled_upper, y_resampled_upper, x_resampled_lower, y_resampled_lower, file_name, _, _, _ = resample_rearrange_santini(
            sant_x, sant_y, sant_files, idx)

        # Pairing up x, y coordinates along the columns of each row first for upper then lower eyelids
        for i in range(len(x_resampled_upper)):
            if i == 0:
                keypoints = [x_resampled_upper[i], y_resampled_upper[i]]
            else:
                keypoints = np.hstack(
                    (keypoints, x_resampled_upper[i], y_resampled_upper[i]))
        for i in range(len(x_resampled_lower)):
            keypoints = np.hstack(
                (keypoints, x_resampled_lower[i], y_resampled_lower[i]))

        if data_folder is not None:
            file_name = 'labeled-data/'+data_folder+'/'+file_name

        # appends the filename before the 64 keypoint resulting in 65 columns
        df_row = np.hstack((file_name, keypoints))

        if flag == False:
            df = df_row
            flag = True
        else:
            df = np.vstack((df, df_row))
        if copy_images:
            #Copying image frames into DLC folder/labeled-data folder
            copyfile(selected_frame, os.path.join(save_folder,data_folder,selected_frame.split('/')[-1]))


    # stacks rows to make the required dataframe adding the 3 headers before the
    df = np.vstack((row1, row2, row3, df))
    pd_df = pd.DataFrame(df)

    if os.path.isfile(os.path.join(save_folder,data_folder,csv_filename)):
        resp = input('File exists, do you want to overwrite? (y/n) \n')
    if os.path.isfile(os.path.join(save_folder,data_folder,csv_filename)) == False or resp == 'y':
        pd.DataFrame.to_csv(pd_df, os.path.join(save_folder,data_folder,csv_filename),
                            header=False, index=False)
    return pd_df

############################################
### ---  Visualization  --- ###
############################################
def viz_selected_frames(save_folder):
    selected_frames_csv = glob.glob(save_folder+'/*.csv')
    assert len(selected_frames_csv) == 1, 'Multiple / no .csv file(s) found in the save folder'
    x_coords, y_coords, file_names = utils.get_human_kpt_labels(selected_frames_csv[0])
    
    for i, png_file in enumerate(file_names):
        frame = cv2.imread(os.path.join(save_folder,png_file.split('/')[-1]))
        print(png_file)
        plt.scatter(x_coords[i], y_coords[i])
        plt.imshow(frame)
        plt.show()
        if i==20:
            print('Visualizing the first 20 frames or less....')
            break
    return True