import os
import cv2
from scipy.spatial import distance
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import imgaug.augmenters as iaa
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from glob import glob
from . import utils


# WIP: add batch size changes to transformations as well
def predict_rsnet_features(frame, model, batch=True, batch_sz=8, multiprocessing=True):
    """Estimate ResNet features for a given frame or batch of frames

    Args:
        frame (array): image frame
        model (keras_model): 

    Returns:
        TYPE: Description
    """
    aug = iaa.Resize({"height": 224, "width": 224})
    img = aug(image=frame)
    trsnfm = image.img_to_array(img)
    trsnfm = np.expand_dims(trsnfm, axis=0)
    trsnfm = preprocess_input(trsnfm)
    if batch:
        trsnfm = model.predict(trsnfm,batch_size=batch_sz, use_multiprocessing=multiprocessing)
    else:
        trsnfm = model.predict(trsnfm,use_multiprocessing=multiprocessing)
    return trsnfm


def get_rnfs_from_list(file_list, model):
    rn_dims = 100352
    _rnfs=np.zeros((len(file_list),rn_dims))
    for i,file in tqdm(enumerate(file_list)):
        _frame = cv2.imread(file)
        _frame_rnfs = predict_rsnet_features(_frame,model,batch=False)
        _rnfs[i,:]=_frame_rnfs.flatten()
    return _rnfs


def select_augmentations(trn_fls, tst_fls, aug_fls,
                         n_frames=498,
                         kmeans_batch_sz=300,
                         kmeans_type='batch',
                         return_min_rand_frames = False,
                         num_augs_per_cnd = 2080,
                         guess_n_clusters=None, 
                         cache_loc = './pylids_cache/'):
    """Summary

    Args:
        train_folder_path (TYPE): Description
        test_folder_path (TYPE): Description
        num_frames (int, optional): Description
        kmeans_batch_size (int, optional): Description
        kmeans_type (str, optional): default, batch and smae_sz
        km_njobs (int, optional): number of threads available to run kmeans same size

    Returns:
        TYPE: Description
    """
    
    print('This can take some time to run, go make yourself some tea!')
    print('\n Run time and RAM required scales with dataset size!')
    print('\n Track memory usage and make sure you have enough RAM for this process.')
    print('\n else, downsample your data using k-means clustering within participants...')
    model = ResNet50(weights="imagenet", include_top=False)
    if not os.path.exists(cache_loc):
        os.makedirs(cache_loc)

    # extracting resnet features for training dataset images
    print('Files in cache: \n')
    print(glob(cache_loc+'*.npy'))
    trn_data = input("Enter the name of the train dataset: \n")
    if os.path.isfile(os.path.join(cache_loc, trn_data+'.npy')):
        print('Loading train features from cache')
        trn_rnfs = np.load(os.path.join(cache_loc, trn_data+'.npy'))
    else:
        trn_rnfs = get_rnfs_from_list(trn_fls, model)
        np.save(os.path.join(cache_loc, trn_data+'.npy'), trn_rnfs)
    if return_min_rand_frames:
        av_trn_fs = np.mean(trn_rnfs,axis=0)
    
    # extracting resnet features for test dataset images
    print('Files in cache: \n')
    print(glob(cache_loc+'*.npy'))
    tst_data = input("Enter the name of the test dataset: \n")
    if os.path.isfile(os.path.join(cache_loc,tst_data+'.npy')):
        print('Loading test features from cache')
        tst_rnfs = np.load(os.path.join(cache_loc,tst_data+'.npy'))
    else:
        tst_rnfs = get_rnfs_from_list(tst_fls, model)
        np.save(os.path.join(cache_loc,tst_data+'.npy'), tst_rnfs)
        
    # Iterative k means which keeps running till we find a given number of frames to label
    # from the test dataset / set of augmented images
    n_clusters = n_frames
    
    #first guess for num of kmeans cluster
    if guess_n_clusters is None:
        tr_ts_ratio = len(trn_fls)/(len(trn_fls)+len(tst_fls))
        cluster_pad = (tr_ts_ratio * n_clusters) / (1 - tr_ts_ratio)
        n_clusters = n_clusters + int(cluster_pad) #first guess
    else:
        n_clusters = guess_n_clusters
    print('Initializing k-means with '+str(n_clusters)+' clusters')
    
    frames_found = 0
    while True:
        print('Running k-means clustering.... \n')
        if kmeans_type =='batch':
            _kmeans = MiniBatchKMeans(n_clusters, random_state=0, batch_size=kmeans_batch_sz).fit(
                np.vstack((trn_rnfs, tst_rnfs)))
        elif kmeans_type == 'default':
            _kmeans = KMeans(n_clusters, random_state=0).fit(
                np.vstack((trn_rnfs, tst_rnfs)))

        print('Calculating number of frames found... \n')
        pwise_dist = distance.cdist(_kmeans.cluster_centers_, np.vstack(((np.mean(trn_rnfs,axis=0)), np.mean(tst_rnfs,axis=0))),'euclidean')
        frames_found = len(np.argwhere(np.argmin(pwise_dist, axis = 1)==1))

        if (frames_found <= n_frames+2) and (frames_found >= n_frames-2):
            print('found '+str(frames_found)+' frames')
            print('Converged!')
            break
        else:
            print('found '+str(frames_found)+' frames')
            print('Did not converge :@, running another iteration... \n')
        if n_frames > frames_found:
            n_clusters = n_clusters + (n_frames-frames_found)
            print('Updating n_clusters to '+str(int(n_clusters)) + '\n')
        else:
            n_clusters = n_clusters - (frames_found-n_frames)
            print('Updating num_clusters to '+str(int(n_clusters)) + '\n')

    tst_km_cntr = np.squeeze(_kmeans.cluster_centers_[np.argwhere(np.argmin(pwise_dist, axis = 1)==1),:])

    del trn_rnfs
    del tst_rnfs
    del pwise_dist
    
    #loading all augmentations(this one is a biggie)
    if os.path.isfile(os.path.join(cache_loc, trn_data+'_augs.npy')):
        print('Loading augmentation features from cache')
        aug_rnfs = np.load(os.path.join(cache_loc, trn_data+'_augs.npy'))
    else:
        aug_rnfs = get_rnfs_from_list(aug_fls, model)
        np.save(os.path.join(cache_loc, trn_data+'_augs.npy'), aug_rnfs)
    
    #find unique frames from kmeans which is closest to augmentations
    print('Selecting augmentations... \n')
    aug_pwise_dist = distance.cdist(tst_km_cntr, aug_rnfs,'cosine')

    km_aug_idxs = []
    for i in range(aug_pwise_dist.shape[0]):
        idx = 0
        while True:
            if np.argsort(aug_pwise_dist[i,:])[idx] in km_aug_idxs:
                idx+=1
            else:
                break
        km_aug_idxs.append(np.argsort(aug_pwise_dist[i,:])[idx])
    km_augs = [aug_fls[i] for i in km_aug_idxs]

    if return_min_rand_frames:
        #selecting random augmentations for every different perturbation
        num_fr_augs = np.zeros(len(aug_fls)//num_augs_per_cnd)
        extra_augs = frames_found%len(num_fr_augs)

        for i in range(len(num_fr_augs)):
            if i < extra_augs:
                num_fr_augs[i] = frames_found//len(num_fr_augs) + 1
            else:
                num_fr_augs[i] = frames_found//len(num_fr_augs)

        rand_aug_idxs = []
        rand_augs = []
        for i, num_fr_aug in enumerate(num_fr_augs):
            rand_aug_idxs = np.random.choice(np.arange(num_augs_per_cnd), size=int(num_fr_aug), replace=False)
            rand_augs = rand_augs + [aug_fls[idx+(num_augs_per_cnd*i)] for idx in rand_aug_idxs]
        
        #selecting augmentations closest to test set
        min_pdist_rn = distance.cdist(av_trn_fs.reshape(1,-1), aug_rnfs, metric='cosine')
        min_aug_idxs = np.argsort(min_pdist_rn[0])[:frames_found]
        min_augs = [aug_fls[idx] for idx in min_aug_idxs]
        
        print('Done!')
        return km_augs, min_augs, rand_augs
    else:
        print('Done!')
        return km_augs


def select_frames_to_label(trn_fls=None, tst_fls=None,
                         n_frames=10,
                         kmeans_batch_sz=300,
                         kmeans_type='batch',
                         return_min_rand_frames = False,
                         cache_loc = './pylids_cache/',
                         load_from_cache = True):
    """Summary

    Args:
        train_folder_path (list, optional): folder path to train frames ()
        test_folder_path (list): folder path to test frames
        num_frames (int, optional): Description
        kmeans_batch_size (int, optional): Description
        kmeans_type (str, optional): default, batch
        load_from_cache (bool, optional): set to True if you want to load/save
                                         trn_fls and tst_fls from cache

    Returns:
        TYPE: Description
    """
    print('This can take some time to run, go make yourself some tea!')
    print('\n Run time and RAM required scales with dataset size!')
    print('\n Track memory usage and make sure you have enough RAM for this process.')
    print('\n else, downsample your data using k-means clustering within participants...')

    model = ResNet50(weights="imagenet", include_top=False)
    if not os.path.exists(cache_loc) and load_from_cache:
        os.makedirs(cache_loc)

    # extracting resnet features for training dataset images
    if trn_fls is not None:
        print('Files in cache: \n')
        print(glob(cache_loc+'*.npy'))
        if load_from_cache:
            trn_data = input("Enter the name of the train dataset: \n")
            if os.path.isfile(os.path.join(cache_loc, trn_data+'.npy')):
                print('Loading train features from cache')
                trn_rnfs = np.load(os.path.join(cache_loc, trn_data+'.npy'))
            else:
                trn_rnfs = get_rnfs_from_list(trn_fls, model)
                np.save(os.path.join(cache_loc, trn_data+'.npy'), trn_rnfs)
        else:
            trn_rnfs = get_rnfs_from_list(trn_fls, model)
        
        if return_min_rand_frames:
            av_trn_fs = np.mean(trn_rnfs,axis=0)
    
    if tst_fls is None:
        assert trn_fls is not None, 'Test files should be provided'
    else:
        # extracting resnet features for test dataset images
        if load_from_cache:
            print('Files in cache: \n')
            print(glob(cache_loc+'*.npy'))
            
            tst_data = input("Enter the name of the test dataset:\n")
            if os.path.isfile(os.path.join(cache_loc,tst_data+'.npy')):
                print('Loading test features from cache')
                tst_rnfs = np.load(os.path.join(cache_loc,tst_data+'.npy'))
            else:
                tst_rnfs = get_rnfs_from_list(tst_fls, model)
                np.save(os.path.join(cache_loc,tst_data+'.npy'), tst_rnfs)
        else:
            tst_rnfs = get_rnfs_from_list(tst_fls, model)

    # Iterative k means which keeps running till we find a given number of frames to label
    # from the test dataset / set of augmented images
    n_clusters = n_frames
    
    if trn_fls is not None:
        #first guess for num of kmeans cluster, to encourage faster convergence
        tr_ts_ratio = len(trn_fls)/(len(trn_fls)+len(tst_fls))
        cluster_pad = (tr_ts_ratio * n_clusters) / (1 - tr_ts_ratio)
        n_clusters = n_clusters + int(cluster_pad) #first guess
        print('Initializing k-means with '+str(n_clusters)+' clusters')
        
        frames_found = 0
        while True:
            print('Running k-means clustering.... \n')
            if kmeans_type =='batch':
                _kmeans = MiniBatchKMeans(n_clusters, random_state=0, batch_size=kmeans_batch_sz).fit(
                    np.vstack((trn_rnfs, tst_rnfs)))
            elif kmeans_type == 'default':
                _kmeans = KMeans(n_clusters, random_state=0).fit(
                    np.vstack((trn_rnfs, tst_rnfs)))

            print('Calculating number of frames found... \n')
            pwise_dist = distance.cdist(_kmeans.cluster_centers_, np.vstack(((np.mean(trn_rnfs,axis=0)), np.mean(tst_rnfs,axis=0))),'euclidean')
            frames_found = len(np.argwhere(np.argmin(pwise_dist, axis = 1)==1))

            if (frames_found <= n_frames+2) and (frames_found >= n_frames-2):
                print('found '+str(frames_found)+' frames')
                print('Converged!')
                break
            else:
                print('found '+str(frames_found)+' frames')
                print('Did not converge :@, running another iteration... \n')
            if n_frames > frames_found:
                n_clusters = n_clusters + (n_frames-frames_found)
                print('Updating n_clusters to '+str(int(n_clusters)) + '\n')
            else:
                n_clusters = n_clusters - (frames_found-n_frames)
                print('Updating num_clusters to '+str(int(n_clusters)) + '\n')

        tst_km_cntr = np.squeeze(_kmeans.cluster_centers_[np.argwhere(np.argmin(pwise_dist, axis = 1)==1),:])

        #find unique frames from test set closest to km centroids
        print('Selecting frames to label... \n')
        km_pwise_dist = distance.cdist(tst_km_cntr, tst_rnfs,'cosine')

        km_frame_idxs = []
        for i in range(km_pwise_dist.shape[0]):
            idx = 0
            while True:
                if np.argsort(km_pwise_dist[i,:])[idx] in km_frame_idxs:
                    idx+=1
                else:
                    break
            km_frame_idxs.append(np.argsort(km_pwise_dist[i,:])[idx])
        km_frames = [tst_fls[i] for i in km_frame_idxs]

    else:
        if kmeans_type =='batch':
            _kmeans = MiniBatchKMeans(n_clusters, random_state=0, batch_size=kmeans_batch_sz).fit(tst_rnfs)
        elif kmeans_type == 'default':
            _kmeans = KMeans(n_clusters, random_state=0).fit(tst_rnfs)

        tst_km_cntr = np.squeeze(_kmeans.cluster_centers_)

        #find unique frames from test set closest to km centroids
        print('Selecting frames to label... \n')
        km_pwise_dist = distance.cdist(tst_km_cntr, tst_rnfs,'cosine')

        km_frame_idxs = []
        for i in range(km_pwise_dist.shape[0]):
            idx = 0
            while True:
                if np.argsort(km_pwise_dist[i,:])[idx] in km_frame_idxs:
                    idx+=1
                else:
                    break
            km_frame_idxs.append(np.argsort(km_pwise_dist[i,:])[idx])
        km_frames = [tst_fls[i] for i in km_frame_idxs]


    if return_min_rand_frames:
        #find unique frames from train set closest to km centroids
        rand_frame_idxs = np.random.choice(np.arange(len(tst_fls)), size=int(frames_found), replace=False)
        rand_frames = [tst_fls[i] for i in rand_frame_idxs]
        
        if trn_fls is not None:
            #selecting augmentations closest to test set
            min_pdist_rn = distance.cdist(av_trn_fs.reshape(1,-1), tst_rnfs, metric='cosine')
            min_frame_idxs = np.argsort(min_pdist_rn[0])[:frames_found]
            min_frames = [tst_fls[idx] for idx in min_frame_idxs]
            print('Done!')
        else:
            min_frames = []
            print('Done!')
        return km_frames, min_frames, rand_frames
    else:
        print('Done!')
        return km_frames


###############################################################
### ---  semi supervised selection of training frames  --- ###
###############################################################

def select_semi_sup_frames(trn_fls, tst_fls, model, n_frames, cache_loc, return_min_rand_frames=False):
    return x,y

def semi_sup_frs_to_dlc_kpts(dlc_vid_ls,
                            dlc_kpt_ls,
                            trn_labeled_data,
                            mode = 'max_lkhood',
                            n_frames = 10,
                            only_eyelids = True,
                            copy_images = False,
                            data_folder= None,
                            save_folder='./',
                            csv_filename='CollectedData_POMlab.csv'):
    """
    Args:
        dlc_vid_ls: list of dlc video filenames
        dlc_kpt_ls: list of dlc keypoint filenames vid_ls and corresponding kpt_ls should be in same order
        trn_labeled_data: list of labeled data
        mode: 'max_lkhood' or 'kmeans'
        n_frames: number of frames to select, if mode is 'kmeans'
        only_eyelids: True or False
        copy_images: True or False
        data_folder: folder where images are stored
        save_folder: folder where dlc kpts are to be saved
        csv_filename: name of csv file to be saved
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

    aug_funcs = [add_defocus_blur, add_exposure, add_gaussian_noise,
    add_jpeg_comp, add_motion_blur, add_mock_glint, add_mock_pupil, add_rotation, add_reflection]
    aug_name = ['dfb', 'exp', 'gno', 'jpg', 'mbl','mkg','mkp','rot','ref']
    refl_files = glob.glob(os.path.join(project_path,'reflection_aug_frames')+'/*.png') #Standardize the location where these frames are stored

    # Main loop
    for dlc_vid, dlc_kpt in zip(dlc_vid_ls, dlc_kpt_ls):
        x,y,c = utils.get_kpts_h5(dlc_kpt)

        #using only pupil keypoints for LPW
        c = c[:,32:]

        #frame with maximum likelihood
        max_conf_fr = np.argmax(np.sum(c,1))
        print('For {} max_lkhood_fr had mean likelihood {}'.format(dlc_vid, np.mean(np.sum(c[max_conf_fr],1))))


        # max_conf_frs = np.argwhere(np.sum(c,1)>likelihood_threshold*c.shape[1])
        # print('Found {} frames with likelihood above {}'.format(len(max_conf_frs),likelihood_threshold))


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
