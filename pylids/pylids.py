from importlib.resources import path
import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from numpy.polynomial import polynomial as P
from skimage.measure import EllipseModel
from skimage.measure import ransac
from scipy.signal import savgol_filter
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import deeplabcut
import appdirs
import tempfile
from . import utils

############################################
### ---  pupil and eylid fits  --- ###
############################################


def ols_pupil_fit(all_points):
    """Least squares pupil fit based on skimage

        The functional model of the ellipse is::
            xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
            yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
            d = sqrt((x - xt)**2 + (y - yt)**2)
        where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
        d is the shortest distance from the point to the ellipse.
        The estimator is based on a least squares minimization. The optimal
        solution is computed directly, no iterations are required. This leads
        to a simple, stable and robust fitting method.
    ----------
    Args:
        all_points (array): (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

    Returns:
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.
    """
    ell_model = EllipseModel()
    ell_model.estimate(all_points)
    xc, yc, a, b, theta = ell_model.params
    return xc, yc, a, b, theta


def fit_pupil(x, y, c=None, p_cutoff=0.3, use_ransac=False, min_samples=5, residual_threshold=1, max_iter=10000):
    """Pupil estimation for a given frame
        given a frames * keypoints array

    Args:
        x (float): x coordinate of keypoint
        y (float): y coordinate of keypoint
        c (float, optional): likelihood or confidence for a keypoint 

        p_cutoff (float, optional): likelihood cutoff for using a given pupil keypoint in the ellipse fit

        use_ransac (bool, optional): Force RANSAC for all pupil estimations, super slow but may be more accurate

        min_samples (int, optional): ransac parameter, minimum random samples
        for fitting ellipses

        residual_threshold (int, optional): ransac parameter
        max_iter (int, optional): ransac parameter

    Returns:
        puil_fits: Dictionary of pupil fit parameters with keys
        pupil_fits['xc'],pupil_fits['yc'],pupil_fits['a'],pupil_fits['b'],pupil_fits['theta']
    """
    xc_arr, yc_arr, a_arr, b_arr, theta_arr = 0, 0, 0, 0, 0

    try:
        #check that 48 keypoints are present
        assert len(x) == 48, 'Found ' + str(len(x)) + ' keypoints, expected 48 (32 for eyelids and 16 for pupil)'
        x_pupil, y_pupil = x[-16:], y[-16:]
        
        if c is None:
            x_pupil_filt, y_pupil_filt = x_pupil, y_pupil
        else:
            x_pupil_filt, y_pupil_filt = x_pupil[c[-16:]
                                                 > p_cutoff], y_pupil[c[-16:] > p_cutoff]
            
        assert len(x_pupil_filt) >= (
            min_samples+1), "Too few keypoints to fit ellipse"

        all_points = np.vstack((x_pupil_filt, y_pupil_filt))
        ell_model = EllipseModel()
        ell_model.estimate(all_points.T)
        res = ell_model.residuals(all_points.T)

        if use_ransac and any(res > 4):
            print('Using RANSAC')
            x_pupil_filt, y_pupil_filt = x_pupil[c[-16:]
                                                 > 0.1], y_pupil[c[-16:] > 0.1]
            all_points = np.vstack((x_pupil_filt, y_pupil_filt))
            ransac_model, inliers = ransac(
                all_points.T, EllipseModel, min_samples, residual_threshold, max_trials=max_iter)  # random state
            res = ransac_model.residuals(all_points.T)
            if any(res > 4):
                pupil_points = np.vstack((x_pupil, y_pupil))
                centroid, label = kmeans2(pupil_points.T, 2, minit='points')
                all_points = np.hstack(
                    (x_pupil[np.argwhere(label == 0)], y_pupil[np.argwhere(label == 0)]))
                ransac_model, inliers = ransac(
                    all_points, EllipseModel, min_samples, residual_threshold, max_trials=max_iter)  # random state
                xc, yc, a, b, theta = ransac_model.params

                all_points = np.hstack(
                    (x_pupil[np.argwhere(label == 1)], y_pupil[np.argwhere(label == 1)]))
                ransac_model, inliers = ransac(
                    all_points, EllipseModel, min_samples, residual_threshold, max_trials=max_iter)  # random state
                xc1, yc1, a1, b1, theta1 = ransac_model.params

                # Selecting the smaller ellipse, rejecting points on Iris
                if ((a-a1)+(b-b1)) < 0:
                    xc_arr, yc_arr, a_arr, b_arr, theta_arr = xc, yc, a, b, theta
                else:
                    xc_arr, yc_arr, a_arr, b_arr, theta_arr = xc1, yc1, a1, b1, theta1

            else:
                xc, yc, a, b, theta = ransac_model.params
                xc_arr, yc_arr, a_arr, b_arr, theta_arr = xc, yc, a, b, theta
        else:
            xc, yc, a, b, theta = ell_model.params
            xc_arr, yc_arr, a_arr, b_arr, theta_arr = xc, yc, a, b, theta
        return xc_arr, yc_arr, a_arr, b_arr, theta_arr
    except:
        xc_arr, yc_arr, a_arr, b_arr, theta_arr = 0, 0, 0, 0, 0
        return xc_arr, yc_arr, a_arr, b_arr, theta_arr


def fit_eyelid(x, y, c,
    deg=4,
    p_cutoff=0.2,
    min_samples=4,
    frame_x_end=400,
    return_full_eyelid=False,
    weighted=True,
    use_constraints=False,
    **kwargs):
    """Weigted OLS on eyelid keypoints to estimate eyelid shape

    Args:
        x (float): x coordinate of keypoint
        
        y (float): y coordinate of keypoint
        
        c (float): likelihood or confidence for a keypoint used as weights for OLS
        
        deg (int, optional): max degree of polynomial for eyelid fit
        
        frame_x_end (int, optional): width of eye image in pixels
        
        return_full_eyelid (bool, optional): Returns x, y coordinates of
        full eyelid, if false returns eye corners and polynomial coefficients
        
        weighted (bool, optional): Weigted polynomial fit for eyelids 

        **kwargs: passed on to utils.parse_keypoints()

    Returns:
        corners_x: [left_corner, right_corner] eye corners the polynomials
        are represntative of the eyelids when evaluated between these two
        x coordinates

        coefs_up, coefs_lo: polynomial coefficiants for upper and lower eyelids

        x_new: the range of x over which the upper and lower eylid polynomials are evaluated

        fit_up, fit_lo: upper and lower eyelids y coordinates
    """
    upper, lower = utils.parse_keypoints(x, y, c, **kwargs)
    eye_lo_x, eye_lo_y, eye_lo_c = lower.T
    eye_up_x, eye_up_y, eye_up_c = upper.T

    x_eye_up_filt, y_eye_up_filt, c_eye_up_filt = eye_up_x[eye_up_c > p_cutoff], eye_up_y[eye_up_c > p_cutoff], eye_up_c[eye_up_c > p_cutoff]
    x_eye_lo_filt, y_eye_lo_filt, c_eye_lo_filt = eye_lo_x[eye_lo_c > p_cutoff], eye_lo_y[eye_lo_c > p_cutoff], eye_lo_c[eye_lo_c > p_cutoff]


    if len(x_eye_up_filt) > (min_samples) and len(x_eye_lo_filt) > (min_samples):
        x_new = np.linspace(np.min(x)-10, np.max(x)+10, num=100)

        if use_constraints:
            polyestimator = polyfit.PolynomRegressor(deg=2)
            convex_constraint = polyfit.Constraints(curvature='convex')
            polyestimator.fit(x_eye_up_filt.reshape(-1,1), y_eye_up_filt, loss = 'l2', constraints={0: convex_constraint})
            coefs_up = polyestimator.coef_
            polyestimator.fit(x_eye_lo_filt.reshape(-1,1), y_eye_lo_filt, loss = 'l2', constraints={0: convex_constraint})
            coefs_lo = polyestimator.coef_
            fit_up = polyestimator.predict(x_new.reshape(-1,1))
            fit_lo = polyestimator.predict(x_new.reshape(-1,1))

        elif weighted:
            coefs_up = P.polyfit(x_eye_up_filt, y_eye_up_filt, deg, full=False, w=c_eye_up_filt)
            coefs_lo = P.polyfit(x_eye_lo_filt, y_eye_lo_filt, deg, full=False, w=c_eye_lo_filt)
            fit_up = P.polyval(x_new, coefs_up)
            fit_lo = P.polyval(x_new, coefs_lo)
            
        else:
            coefs_up = P.polyfit(x_eye_up_filt, y_eye_up_filt, deg, full=False)
            coefs_lo = P.polyfit(x_eye_lo_filt, y_eye_lo_filt, deg, full=False)
            fit_up = P.polyval(x_new, coefs_up)
            fit_lo = P.polyval(x_new, coefs_lo)

        intersect_x, intersect_y = utils.intersection(x_new, fit_up, x_new, fit_lo)

        frame_x_str = 0
        if intersect_x.shape[0] != 0:
            l_crnr = []
            r_crnr = []
            for int_x in intersect_x:
                if (frame_x_str-100) < int_x < (frame_x_str+100):
                    l_crnr.append(int_x)
                if (frame_x_end-100) < int_x < (frame_x_end+100):
                    r_crnr.append(int_x)
            if l_crnr != [] and r_crnr != []:
                l_crnr = min(l_crnr)
                r_crnr = max(r_crnr)
                x_new = np.linspace(l_crnr, r_crnr, num=100)
            elif l_crnr == [] and r_crnr != []:
                r_crnr = max(r_crnr)
                x_new = np.linspace(np.min(x)-10, r_crnr, num=100)
            elif l_crnr != [] and r_crnr == []:
                l_crnr = min(l_crnr)
                x_new = np.linspace(l_crnr, np.max(x)+10, num=100)
            fit_up = P.polyval(x_new, coefs_up)
            fit_lo = P.polyval(x_new, coefs_lo)

        corners_x = [x_new[0], x_new[-1]]

        if return_full_eyelid:
            return x_new, fit_up, fit_lo, corners_x, coefs_up, coefs_lo
        else:
            return x_new, fit_up, fit_lo
    else:
        x_new, fit_up, fit_lo, corners_x, coefs_up, coefs_lo = 0, 0, 0, 0, 0, 0
        if return_full_eyelid:
            return x_new, fit_up, fit_lo, corners_x, coefs_up, coefs_lo
        else:
            return x_new, fit_up, fit_lo


############################################
### ---  DLC wrapper  --- ###
############################################


def dlc_estimate_kpts(eye_vid, eye_id, path_config_file, save_dlc_output, dest_folder, batch_sz, estimate_pupils, estimate_eyelids):
    if eye_id == 1:
        if save_dlc_output:
            deeplabcut.analyze_videos(path_config_file,
                                        [eye_vid],
                                    videotype='.mp4',
                                    batchsize=batch_sz, #change in pose_config.yml to 1
                                    destfolder=dest_folder)
            assert len(glob.glob(os.path.join(dest_folder,'*.h5')))==1, 'Total config files in ' + str(tmpdirname) + ' is not equal to 1'
            kpt_h5 = glob.glob(os.path.join(dest_folder,'*.h5'))[0]
            x, y, c = utils.get_kpts_h5(kpt_h5)

        
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                deeplabcut.analyze_videos(path_config_file,
                                            [eye_vid],
                                        videotype='.mp4',
                                        batchsize=batch_sz, #change in pose_config.yml to 1
                                        destfolder=tmpdirname)
                print(glob.glob(os.path.join(tmpdirname,'*.h5')))
                assert len(glob.glob(os.path.join(tmpdirname,'*.h5')))==1, 'Total config files in ' + str(tmpdirname) + ' is not equal to 1'
                kpt_h5 = glob.glob(os.path.join(tmpdirname,'*.h5'))[0] #or filtered .h5
                x, y, c = utils.get_kpts_h5(kpt_h5)

    else:
        if save_dlc_output:
            deeplabcut.analyze_videos(path_config_file,
                                        [eye_vid],
                                    videotype='.mp4',
                                    batchsize=batch_sz,
                                    destfolder=dest_folder,
                                    flip_video = True)
            assert len(glob.glob(os.path.join(dest_folder,'*.h5')))==1, 'Total config files in ' + str(tmpdirname) + ' is not equal to 1'
            kpt_h5 = glob.glob(os.path.join(dest_folder,'*.h5'))[0]
            x, y, c = utils.get_kpts_h5(kpt_h5)

        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                deeplabcut.analyze_videos(path_config_file,
                                            [eye_vid],
                                        videotype='.mp4',
                                        batchsize=batch_sz, 
                                        destfolder=tmpdirname,
                                        flip_video = True)
                print(glob.glob(os.path.join(tmpdirname,'*.h5')))
                assert len(glob.glob(os.path.join(tmpdirname,'*.h5')))==1, 'Total config files in ' + str(tmpdirname) + ' is not equal to 1'
                kpt_h5 = glob.glob(os.path.join(tmpdirname,'*.h5'))[0]
                x, y, c = utils.get_kpts_h5(kpt_h5)
    return x, y, c

############################################
### ---  main wrapper function  --- ###
############################################
def analyze_video(eye_vid=None,
                    model_name='deploy_pupils',
                    batch_sz =8,
                    eye_id = None,
                    estimate_pupils=True,
                    estimate_eyelids=True,
                    use_ransac=False,
                    timestamp_file=None,
                    dest_folder='./',
                    npz_file= None,
                    out_file='eye_annotation.mp4',
                    progress_bar=None,
                    constraint_eyefit=False,
                    save_dlc_output=False,
                    save_vid=False,
                    save_npz=False,
                    annot_lw=2, annot_clr=(0, 0, 255)):
    ''' 
    Use pylids to analyze an eye video and return pupil and eyelid positions.
    Parameters
    ----------
    eye_vid : str
        Path to video file to analyze

    model_name : str
        Name of the model to use for pupil and eyelid estimation.
        For available models, use function pylids.available_models()

        One can also directly use a locally trained model by providing
        the path to the model config.yaml file as the model_name argument.

    batch_sz : int
        Batch size for DLC - reduce this if your graphics card runs out
        of memory. Default set based on RTX 2080Ti.
    
    eye_id : int
        0 for right eye (inverted video feed), 1 for left eye (upright video feed)
        based on Pupil Labs convention.

    estimate_pupils : bool
        Estimate pupil positions, for speedup can set to False if estimating only eyelids

    estimate_eyelids : bool
        Estimate eyelid positions, for speedup can set to False if estimating only pupils

    use_ransac : bool
        Use RANSAC to filter pupil positions, 
        slows down analysis but can be more robust to outliers

    timestamp_file : str
        Path to timestamp file (relavant for VEDB dataset, can ignore otherwise)

    dest_folder : str
        Path to destination folder for saving all outputs

    npz_file : str
        Path to npz file to save eyelid and pupil positions

    out_file : str
        Path to output video file if save_vid is True

    progress_bar : function
        Function to display progress bar (variants of tqdm)

    constraint_eyefit : bool WIP!!!!
        Constrains the eyelid fits to be concave and convex for the upper and lower eyelids respectively
        EXPERIMENTAL - may not work well for all eyes.

    save_dlc_output : bool
        If True saves DLC keypoint outputs to dest_folder,
        this is useful for debugging and for re-running the 
        analysis with different fit parameters

    save_vid : bool
        Save annotated video to out_file
    
    save_npz : bool
        Save pupil and eyelid positions to npz_file in pupil labs format
        If False only returns the positions as a dictionary
    
    annot_lw : int
        Line width for annotated video
    
    annot_clr : tuple
        Color for annotated video, default is red
    '''
    
    # looking for timestamps, specific to the vedb project
    if timestamp_file is not None:
        timestamps = np.load(timestamp_file)

    if progress_bar is None:
        def progress_bar(x): return x

    # checking if left or right eye is loaded
    if eye_id is None:
        if 'eye0' in eye_vid:
            eye_id = 0
        elif 'eye1' in eye_vid:
            eye_id = 1
        else:
            raise ValueError("If video is not `eye0.mp4` or eye1.mp4`, per pupil labs conventions, then eye_id kwarg must be specified! use eye_id = 1 for upright videos and eye_id = 0 for inverted videos.")
    #if user provided locally trained model
    if model_name.endswith('.yaml'):
        path_config_file = model_name
        assert os.path.isfile(path_config_file), model_name + ' config.yaml file not found'
    #else download weights and use model from pylids
    else:
        # download dlc model weights into pylids config folder, if they do not already exist
        utils.get_model_weights(model=model_name)

        # points to appropriate config file based on model_name arg supplied 
        usr_config_path = appdirs.user_config_dir()
        path_config_file = os.path.join(usr_config_path,'pylids',model_name,'config.yaml')
        assert os.path.isfile(path_config_file), model_name + ' config.yaml file not found'
    
    #a wrapper function which runs DLC to estimate keypoints
    x,y,c = dlc_estimate_kpts(eye_vid, eye_id, path_config_file, save_dlc_output, dest_folder, batch_sz, estimate_pupils, estimate_eyelids)
    
    # main outputs are saved as a list of dicts
    dlc_dicts = []

    # read a frame to estimate frame size
    vid = cv2.VideoCapture(eye_vid)
    _, frame = vid.read()
    
    if not save_vid:
        vid.release()
    # setup for creating pylids output video
    else:
        video = cv2.VideoWriter(os.path.join(dest_folder, out_file), cv2.VideoWriter_fourcc(
            *"mp4v"), 120, (frame.shape[1], frame.shape[0]))

        l_thick = annot_lw
        is_closed = True
        color = annot_clr
        vid.set(1, 0)

    # main loop iterate over all frames
    for i in progress_bar(range(x.shape[0])):
        # select keypoint from the current frame
        if eye_id == 1:
            x_fr, y_fr, c_fr = x[i, :], y[i, :], c[i, :]
        else:
            x_fr, y_fr, c_fr = x[i, :], frame.shape[0]-y[i, :], c[i, :]
        # frame level output dict
        dlc_annot ={}
        # saving original dlc keypoints
        dlc_annot['dlc_kpts_x'] = x_fr
        dlc_annot['dlc_kpts_y'] = y_fr
        dlc_annot['dlc_confidence'] = c_fr

        if estimate_eyelids:
            # estimate eyelid shape
            x_viz_eye, fit_eye_up, fit_eye_lo = fit_eyelid(x_fr, y_fr, c_fr, return_full_eyelid=False,frame_x_end=frame.shape[1],use_constraints=constraint_eyefit)
            dlc_annot['eyelid_up_y'] = fit_eye_up
            dlc_annot['eyelid_lo_y'] = fit_eye_lo
            dlc_annot['eyelid_x'] = x_viz_eye
            if estimate_eyelids and not estimate_pupils:
                # to make sure vedb-gaze does not mark this session as failed
                dlc_annot['norm_pos'] = [0,0]
            
        if estimate_pupils:
            # estimate pupil position           
            el_xc, el_yc, el_a, el_b, el_theta = fit_pupil(x_fr, y_fr, c_fr)
            #as per pupil labs conventions
            dlc_ellipse = {}
            dlc_ellipse['axes'] = [el_a*2, el_b*2]
            dlc_ellipse['angle'] = el_theta * (180/np.pi)
            dlc_ellipse['center'] = [el_xc, el_yc]
            dlc_annot['ellipse'] = dlc_ellipse

            dlc_annot['confidence'] = np.mean(c_fr[32:]) #FLAG
            dlc_annot['norm_pos'] = [ el_xc/frame.shape[1],  el_yc/frame.shape[0]]
            dlc_annot['diameter'] =  np.mean([el_a*2, el_b*2])

        dlc_annot['id'] = eye_id
        if timestamp_file is not None:
            dlc_annot["timestamp"]=timestamps[i]
        
        # append frame output to main list of dicts
        dlc_dicts.append(dlc_annot)

        # create pylids annotation video
        if save_vid:
            _, frame = vid.read()
            #WIP
            # if eye_id == 0:
            #     frame = cv2.flip(frame, 0)

            if estimate_eyelids:
                corner = [0, 0]
                if x_viz_eye != 0: #No eyelid detected
                    for k in range(len(x_viz_eye)):
                        corner = np.vstack((corner, [x_viz_eye[k], fit_eye_up[k]]))
                    for k in range(len(x_viz_eye)):
                        corner = np.vstack(
                            (corner, [x_viz_eye[-(k+1)], fit_eye_lo[-(k+1)]]))
                    corner = corner[1:]
                    corner = corner.reshape((-1, 1, 2))

                    # draw eyelids
                    frame = cv2.polylines(frame, np.int32([corner]),
                                        is_closed, color, l_thick)

            if estimate_pupils:
                if el_xc != 0: #No pupil detected
                    center_coords = (int(el_xc), int(el_yc))
                    axes_l = (int(el_a), int(el_b))
                    strt = 0
                    end = 360

                    # draw pupil
                    frame = cv2.ellipse(frame, center_coords, axes_l,
                                        (el_theta*180/np.pi), strt, end, color, l_thick)
            video.write(frame)            

    if save_vid:
        vid.release()
        video.release() 
        print('Analyzed video saved in ' + os.path.join(dest_folder,out_file))

    #Convert output to pupil labs format
    dlc_dicts_out = utils.dictlist_to_arraydict(dlc_dicts)
    if save_npz:
        if npz_file is None:
            npz_file = 'dlc_annotation.npz'
        np.savez(os.path.join(dest_folder, npz_file), **dlc_dicts_out)
        print('Analyzed eye video saved in' + os.path.join(dest_folder, npz_file))
    
    return dlc_dicts_out


############################################
### --- utility functions  --- ###
############################################

def available_models():
    usr_config_dir = appdirs.user_config_dir()
    wt_fl = open(os.path.join(usr_config_dir,'pylids','weights_index.txt')).read()
    wts = wt_fl.split('\n')
    if len(wts[-1]) == 0: 
        # clip last line if just EOF
        wts = wts[:-1]

    model_names = []
    for wt in wts:
        model_name, model_url = wt.split(',')
        model_names.append(model_name)
    print('Available models\n')
    print(model_names)
