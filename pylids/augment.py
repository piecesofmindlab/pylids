import os
import cv2
import glob
import math
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2

############################################
### --- image augmentation functions --- ###
############################################


def add_exposure(frame, it=4, batch=True, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Add exposure to the frame, simulate the effect of a camera with a different exposure, for example when recording in 
    bright sunlight.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the exposure. Suggested values are between 0 and 5.
    batch : bool
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.
    '''
    aug = iaa.AddToBrightness((it+1)*30)
    if batch:
        image_aug = aug(images=frame)
        if save:
            for image in image_aug:
                cv2.imwrite(os.path.join(dest_folder, save_file), image)
    else:
        image_aug = aug(image=frame)
        if save:
            cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)

    return image_aug

# JPEG compression


def add_jpeg_comp(frame, it=4, batch=True, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Simulate the effect of videos with different compression rates.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the exposure. Suggested values are between 0 and 5.
    batch : bool
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.
    '''
    aug = iaa.JpegCompression(compression=(it+1)*8+60)
    if batch:
        image_aug = aug(images=frame)
        if save:
            for image in image_aug:
                cv2.imwrite(os.path.join(dest_folder, save_file), image)
    else:
        image_aug = aug(image=frame)
        if save:
            cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)

    return image_aug

# Defocus blur


def add_defocus_blur(frame, it=4, batch=True, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Simulate the effect of a defocus blur.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool
        If True, the frame is a list of frames.
    save_file : str
        file name for saving.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.
    '''
    aug = iaa.imgcorruptlike.DefocusBlur(severity=it+1)
    if batch:
        image_aug = aug(images=frame)
        if save:
            for image in image_aug:
                cv2.imwrite(os.path.join(dest_folder, save_file), image)
    else:
        image_aug = aug(image=frame)
        if save:
            cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)

    return image_aug

# Save fixed frames for this from the dashcam video
# Reflection


def reflection_blending(alpha, beta, e, r):
    e = e.flatten()
    r = r.flatten()
    reflFrame = np.zeros(len(e.flatten()))
    gamma = np.maximum(np.minimum(0, r/255+e/255-alpha), beta)
    reflFrame = e+(gamma*r*(255-e))/255
    return np.uint8(reflFrame.reshape(480, 640))


def add_reflection(frame, reflectionFrame, it=4, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Blend two images to simulate effect of reflection.
    Implemented as discussed in [1]

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    reflectionFrame : numpy.ndarray
        The reflected frame.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool  #Not implemented
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are savedata = kp_df.to_numpy().

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.

    References
    ----------
    [1] Eivazi, S., Santini, T., Keshavarzi, A., Kübler, T., & Mazzei, A.
    (2019, June). Improving real-time CNN-based pupil detection through
    domain-specific data augmentation. In Proceedings of the 11th ACM 
    Symposium on Eye Tracking Research & Applications (pp. 1-6).

    '''

    alpha = 0.5
    beta = 0.10
    e = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(reflectionFrame, cv2.COLOR_BGR2GRAY)
    aug = iaa.Resize({"height": 480, "width": 640})  # remove magic number
    r = aug(image=r)
    image_aug = reflection_blending(alpha, (it+1)*beta, e, r)
    if save:
        cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)
    return image_aug


def add_mock_pupil(frame, it=4, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Adds fake pupil like structures to eye images by creating blacked out
    ellipses.Inspired by [1].

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool  #Not implemented
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.

    References
    ----------
    [1] Eivazi, S., Santini, T., Keshavarzi, A., Kübler, T., & Mazzei, A.
    (2019, June). Improving real-time CNN-based pupil detection through
    domain-specific data augmentation. In Proceedings of the 11th ACM 
    Symposium on Eye Tracking Research & Applications (pp. 1-6).
    '''
    scale_arr = np.linspace(0, 10, 4)
    # vary x and y using a Gaussian distribution centered on the middle of the
    y_arr = np.linspace(0, frame.shape[0], frame.shape[0])
    x_arr = np.linspace(0, frame.shape[1], frame.shape[1])
    x_cent, y_cent = int(
        np.ceil(frame.shape[0]/2)), int(np.ceil(frame.shape[1]/2))
    x, y = np.meshgrid(x_arr, y_arr)
    theta = np.random.uniform(low=0, high=np.pi)
    x0 = np.random.normal(loc=y_cent, scale=100)
    # x center, half width
    a = 5 + scale_arr[it] + np.abs(np.random.normal(scale=30))
    y0 = np.random.normal(loc=x_cent, scale=100)
    # y center, half height
    b = 5 + scale_arr[it] + np.abs(np.random.normal(scale=30))

    ellipse = ((((x-x0)*np.cos(theta)+(y-y0)*np.sin(theta))/a)**2
               + (((x-x0)*np.sin(theta)-(y-y0)*np.cos(theta))/b)**2) <= 1  # True for points inside the ellipse
    idx = np.argwhere(ellipse == True)
    frame[idx[:, 0], idx[:, 1], :] = np.random.uniform(low=0, high=50)
    if save:
        cv2.imwrite(os.path.join(dest_folder, save_file), frame)
    return frame


def add_mock_glint(frame, it=4, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Adds fake glints to eye images to simulate corneal reflection of
    sulight. Created by adding white ellipses. Inspired by [1].

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool  #Not implemented
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.

    References
    ----------
    [1] Eivazi, S., Santini, T., Keshavarzi, A., Kübler, T., & Mazzei, A.
    (2019, June). Improving real-time CNN-based pupil detection through
    domain-specific data augmentation. In Proceedings of the 11th ACM 
    Symposium on Eye Tracking Research & Applications (pp. 1-6).
    '''
    scale_arr = np.linspace(0, 10, 4)
    # vary x and y using a Gaussian distribution centered on the middle of the
    y_arr = np.linspace(0, frame.shape[0], frame.shape[0])
    x_arr = np.linspace(0, frame.shape[1], frame.shape[1])
    x_cent, y_cent = int(
        np.ceil(frame.shape[0]/2)), int(np.ceil(frame.shape[1]/2))
    x, y = np.meshgrid(x_arr, y_arr)
    theta = np.random.uniform(low=0, high=np.pi)
    x0 = np.random.normal(loc=y_cent, scale=50)
    # x center, half width
    a = 5 + scale_arr[it] + np.abs(np.random.normal(scale=30))
    y0 = np.random.normal(loc=x_cent, scale=50)
    # y center, half height
    b = 5 + scale_arr[it] + np.abs(np.random.normal(scale=30))
    ellipse = ((((x-x0)*np.cos(theta)+(y-y0)*np.sin(theta))/a)**2
               + (((x-x0)*np.sin(theta)-(y-y0)*np.cos(theta))/b)**2) <= 1  # True for points inside the ellipse
    idx = np.argwhere(ellipse == True)
    frame[idx[:, 0], idx[:, 1], :] = 255
    if save:
        cv2.imwrite(os.path.join(dest_folder, save_file), frame)
    return frame


def add_gaussian_noise(frame, it, batch=True, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Adds fake glints to eye images to simulate corneal reflection of
    sulight. Created by adding white ellipses. Inspired by [1].

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool  #Not implemented
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.

    References
    ----------
    [1] Eivazi, S., Santini, T., Keshavarzi, A., Kübler, T., & Mazzei, A.
    (2019, June). Improving real-time CNN-based pupil detection through
    domain-specific data augmentation. In Proceedings of the 11th ACM 
    Symposium on Eye Tracking Research & Applications (pp. 1-6).
    '''
    scale_arr = np.linspace(0.1*255, 0.3*255, 4)
    aug = iaa.AdditiveGaussianNoise(scale=scale_arr[it])
    if batch:
        image_aug = aug(images=frame)
        if save:
            for image in image_aug:
                cv2.imwrite(os.path.join(dest_folder, save_file), image)
    else:
        image_aug = aug(image=frame)
        if save:
            cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)

    return image_aug


def rotated_rect_with_max_area(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.

    adapted from: 
    """
    angle = np.deg2rad(angle)
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    return wr, hr


def add_rotation(frame, key_pts=None, it=4, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Simulates a rotated image. Part of the image which go outside the original
    frame are cropped and image is resized to be the same size are original.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool  #Not implemented
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.
    '''
    theta = (it+1)*2.5
    rep = rotated_rect_with_max_area(frame.shape[1], frame.shape[0], theta)
    seq = iaa.Sequential([
        iaa.Rotate(theta),
        iaa.CenterCropToFixedSize(height=int(rep[1]), width=int(rep[0])),
        iaa.Resize({"height": frame.shape[0], "width": frame.shape[1]})
    ])
    if key_pts != None:
        image_aug, kpts_aug = seq(image=frame, keypoints=key_pts)
        return image_aug, kpts_aug
    else:
        image_aug = seq(image=frame)
    if save:
        cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)

        return image_aug


def add_motion_blur(frame, it=4, batch=True, save=False, save_file='augmented_im.png', dest_folder='./'):
    '''
    Adds motion blur to image to simulate the effect of blinks and saccades.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame to be augmented.
        or a list of frames if batch = True
    it : int
        Intensity of the defocus blur. Suggested values are between 0 and 5.
    batch : bool 
        If True, the frame is a list of frames.
    save : bool
        If True, the frame/frames are saved in the dest_folder.
    save_file : str
        file name for saving.
    dest_folder : str
        The folder where the frame/frames are saved.

    Returns
    -------
    img_aug : numpy.ndarray
        The augmented frame / frames.
    '''
    # k values from 30 to 90 with 20 space
    scale_arr = np.linspace(20, 80, 4)
    aug = iaa.MotionBlur(k=int(scale_arr[it]), angle=(-45, 45), direction=1)
    if batch:
        image_aug = aug(images=frame)
        if save:
            for image in image_aug:
                cv2.imwrite(os.path.join(dest_folder, save_file), image)
    else:
        image_aug = aug(image=frame)
        if save:
            cv2.imwrite(os.path.join(dest_folder, save_file), image_aug)

    return image_aug


def generate_all_augmentations(project_path, dataset_name,trn_files):
    main_folder = os.path.join(project_path,dataset_name)
                               
    if os.path.isdir(main_folder) == False: 
        os.mkdir(main_folder)
    else:
        resp = input('Augmentations for given dataset already exists, do you want to overwrite (y/n)? \n')
        if resp == 'y': 
            pass
        else:
            return print('aborted by user')
        
    aug_funcs = [add_defocus_blur, add_exposure, add_gaussian_noise,
            add_jpeg_comp, add_motion_blur, add_mock_glint, add_mock_pupil, add_rotation, add_reflection]
    aug_name = ['dfb', 'exp', 'gno', 'jpg', 'mbl','mkg','mkp','rot','ref']
    aug_func_batch = [add_defocus_blur, add_exposure, add_gaussian_noise,
            add_jpeg_comp, add_motion_blur]
    refl_files = glob.glob(os.path.join(project_path,'reflection_aug_frames')+'/*.png') #Standardize the location where these frames are stored
    images = load_batch(trn_files)
    images = images.astype('uint8')
    for idx, aug_func in enumerate(aug_funcs):
        save_folder = os.path.join(main_folder,aug_name[idx])

        if os.path.isdir(os.path.join(main_folder,aug_name[idx])) == False:
            os.mkdir(os.path.join(main_folder,aug_name[idx]))

        for intensity in range(4):
            if aug_func in aug_func_batch:
                images_aug = aug_func(images, it=intensity, batch = True, save=False)
                for i, trn_file in enumerate(trn_files):
                    sub_num = trn_file.split('/')[-2].split('_')[0]
                    file_name = aug_name[idx]+str(intensity)+'_'+sub_num+'_'+trn_file.split('/')[-1]
                    cv2.imwrite(os.path.join(save_folder,file_name),images_aug[i,:,:,:])
            else:
                for i, trn_file in enumerate(trn_files):
                    image = cv2.imread(trn_file)
                    if aug_func == add_reflection:
                        reflectionIm = cv2.imread(refl_files[np.random.randint(0,len(refl_files))])
                        image_aug = aug_func(image, reflectionIm, it=intensity, save=False)
                    elif aug_func == add_rotation:
                        image_aug, kps_ = aug_func(image, key_pts = [],  it=intensity, save=False)
                    else:
                        image_aug = aug_func(image, it=intensity, save=False)

                    sub_num = trn_file.split('/')[-2].split('_')[0]
                    file_name = aug_name[idx]+str(intensity)+'_'+sub_num+'_'+trn_file.split('/')[-1]
                    cv2.imwrite(os.path.join(save_folder,file_name), image_aug)
    print('All augmentations saved to ' + main_folder)


def list_to_batch(l, batch_size=8):
    """
    Divide a list into chunks of size n.
    """
    new_list = []
    for i in range(0, len(l),batch_size):
        new_list.append(l[i:i + batch_size])
    return new_list


def load_batch(file_list):
    frame = cv2.imread(file_list[0])
    batch_array = np.zeros((len(file_list), frame.shape[0], frame.shape[1], 3))
    for i, file in enumerate(file_list):
        batch_array[i, :, :, :] = cv2.imread(file)
    return batch_array


# Converts dlc keypoints to format required by imgaug
def conv_to_iaug_kpts(x_labels, y_labels, frame):
    kps = KeypointsOnImage.from_xy_array(
        np.vstack((x_labels, y_labels)).T, shape=frame.shape)
    return kps

def conv_to_dlc_kpts(kpts): #what to do with points out of image? Replace with nan
    # keypoints = kpts.clip_out_of_image()
    keypoints = kpts.to_xy_array()
    keypoints = keypoints.flatten()
    return keypoints
