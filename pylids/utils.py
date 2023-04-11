import os
import numpy as np
import pandas as pd
import requests
import zipfile
import appdirs
import deeplabcut
import tempfile

############################################
### ---  parsing dlc keypoints  --- ###
############################################

def parse_keypoints(dlc_keypoints_x, dlc_keypoints_y, dlc_confidence=None,
                    n_corners=2, 
                    n_keypoints_per_lid=15,
                    n_pupil=16,
                   ):
    """parse arrays of labeled keypoints from dlc into more clearly labeled arrays

    Parameters
    ----------
    dlc_keypoints_x : array-like
        list of X coordinates for all labeled keypoints
    dlc_keypoints_y : array-like
        list of Y coordinates for all labeled keypoints
    dlc_confidence : array-like
        list of confidence values for all labeled keypoints
    n_corners : int, optional
        number of corner points (this may be useless as input, should
        always be 2 probably), by default 2
    n_keypoints_per_lid : int, optional
        number of labeled keypoints per lid, by default 15
    n_pupil : int, optional
        number of labeled keypoints around the pupil, by default 16

    Returns
    -------
    upper, lower
        (n_frames x 3) arrays of keypoints for upper and lower lids; 3
        dimensions are X, Y, confidence
    """
    if dlc_confidence is not None:
        corner_pts = np.vstack([dlc_keypoints_x[:n_corners], 
                                dlc_keypoints_y[:n_corners],
                            dlc_confidence[:n_corners],
                            ]
                            ).T
        upper = np.vstack([dlc_keypoints_x[n_corners:(n_corners+n_keypoints_per_lid)],
                        dlc_keypoints_y[n_corners:(n_corners+n_keypoints_per_lid)],
                        dlc_confidence[n_corners:(n_corners+n_keypoints_per_lid)],
                        ]).T
        lower = np.vstack([dlc_keypoints_x[(n_corners+n_keypoints_per_lid):(n_corners+n_keypoints_per_lid * 2)],
                        dlc_keypoints_y[(n_corners+n_keypoints_per_lid):(n_corners+n_keypoints_per_lid * 2)],
                        dlc_confidence[(n_corners+n_keypoints_per_lid):(n_corners+n_keypoints_per_lid * 2)],
                        ]).T
    else:
        corner_pts = np.vstack([dlc_keypoints_x[:n_corners], 
                                dlc_keypoints_y[:n_corners],
                                ]).T
        upper = np.vstack([dlc_keypoints_x[n_corners:(n_corners+n_keypoints_per_lid)],
                        dlc_keypoints_y[n_corners:(n_corners+n_keypoints_per_lid)],
                        ]).T
        lower = np.vstack([dlc_keypoints_x[(n_corners+n_keypoints_per_lid):(n_corners+n_keypoints_per_lid * 2)],
                        dlc_keypoints_y[(n_corners+n_keypoints_per_lid):(n_corners+n_keypoints_per_lid * 2)],
                        ]).T
    upper = np.vstack([corner_pts, upper])
    lower = np.vstack([corner_pts, lower])
    return upper, lower


def get_kpts_h5(h5_file): #WIP change to get_dlc_kpts_h5 and .csv
    """Convert DLC keypoints in .h5 file to array

    Args:
        h5_file (string): Path to DLC .h5 file

    Returns:
        x,y,c: x,y coordinates and c (likelihood) for each keypoint in the image
    """
    kp_df = pd.read_hdf(h5_file)
    data = kp_df.to_numpy()
    x, y, c = data[:, ::3], data[:, 1::3], data[:, 2::3]
    return x, y, c


def get_kpts_csv(csv_file):
    """Convert DLC keypoints in .csv file to array

    Args:
        csv_file (string): Path to DLC .csv file

    Returns:
        x,y,c: x,y coordinates and c (likelihood) for each keypoint in the image
    """
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=3)
    x, y, c = data[:, 1::3], data[:, 2::3], data[:, 3::3]
    return x, y, c


def get_human_kpt_labels(csv_file): #Add only eyelids option
    """Read labels from hand labeled eye images and convert to array

    Args:
        csv_file (string): Path to human labeled .csv file

    Returns:
        x_tru, y_tru, file_names: x,y coordinates and files names for each eye image
    """
    csv_df = pd.read_csv(csv_file, skiprows=2)
    file_names = csv_df['coords']
    csv_data = np.genfromtxt(csv_file, delimiter=',', skip_header=True)
    x_tru, y_tru = csv_data[2:, 1::2], csv_data[2:, 2::2]
    return x_tru, y_tru, file_names

# WIP
# def parse_kpts(h5_file):
# Convert keypoints to a dict with upper lower eyelid and pupil


def get_model_weights(model):
    usr_config_dir = appdirs.user_config_dir()
    if not os.path.exists(os.path.join(usr_config_dir, 'pylids', model)):
        wt_fl = open(os.path.join(usr_config_dir,'pylids','weights_index.txt')).read()
        wts = wt_fl.split('\n')
        if len(wts[-1]) == 0: 
            # clip last line if just EOF
            wts = wts[:-1]

        model_names = []
        for wt in wts:
            model_name, model_url = wt.split(',')
            model_names.append(model_name)
            if model_name == model:
                break

        assert model in model_names, "model does not exist \n available models are \n" +str(model_names) + '\n You selected: ' + model

        print("Downloading model weights for model " + model)
    
        r = requests.get(model_url)

        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname,model +'.zip'),'wb') as f:
                f.write(r.content)
            with zipfile.ZipFile(os.path.join(tmpdirname,model+'.zip')) as z:
                z.extractall(os.path.join(usr_config_dir, 'pylids'))

        path_config = os.path.join(usr_config_dir,'pylids',model,'config.yaml')
        cfg=deeplabcut.auxiliaryfunctions.read_plainconfig(path_config)
        cfg['project_path'] = os.path.join(usr_config_dir,'pylids',model_name)
        deeplabcut.auxiliaryfunctions.write_plainconfig(path_config,cfg)

        trainposeconfigfile,_,_=deeplabcut.return_train_network_path(path_config, shuffle=1)
        cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
        cfg_dlc['project_path'] = os.path.join(usr_config_dir,'pylids', model_name)
        deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

        print("Done! \n")
    else:
        print('Model weights already exist!')    


############################################
### ---  VEDB-gaze utils --- ###
############################################

def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays

        adapted from: www.github.com/vedb/vedb-gaze
    """
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts
    
        adapted from: www.github.com/vedb/vedb-gaze
    """
    dict_fields = list(arraydict.keys())
    first_key = dict_fields[0]
    n = len(arraydict[first_key])
    out = []
    for j in range(n):
        frame_dict = {}
        for k in dict_fields:
            value = arraydict[k][j]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            frame_dict[k] = value
        out.append(frame_dict)
    return out

    #MISC


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4

    
def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]