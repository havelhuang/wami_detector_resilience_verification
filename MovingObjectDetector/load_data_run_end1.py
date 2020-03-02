import numpy as np
import cv2
import timeit
import hdf5storage
import math
import TrainNetwork.TN_BaseFunctions as basefunctions
import pickle
from copy import copy
from copy import deepcopy as deepcopy
import matplotlib.pyplot as plt
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
from SimpleTracker.KalmanFilter import KalmanFilter
from SimpleTracker.KalmanFilter_attack import KalmanFilter_attack
from MovingObjectDetector.MOD_BaseFunctions import TimePropagate, TimePropagate_, draw_error_ellipse2d
from MovingObjectDetector.Init_Track_From_Groundtruth import init_Track_From_Groundtruth

class location:
    def __init__(self, x, y, delta=None, points=None):
        self.x = x
        self.y = y
        self.delta = delta
        self.points = points


def distance(loc1, loc2):
    x_diff = loc1[0] - loc2[0]
    y_diff = loc1[1] - loc2[1]
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)


## to measure the difference between two tracks
## each track is a vector of locations
def diff_mean(track1, track2):
    n = len(track1)
    res = 0
    for i in range(0, n):
        res += distance(track1[i], track2[i])
    return res * 1.0 / n


def diff_max(track1, track2):
    n = len(track1)
    res = 0
    for i in range(0, n):
        tmp = distance(track1[i], track2[i])
        if res < tmp:
            res = tmp
    return res
####


def load_data_run_end1(load_data, attack, model_folder, imagefolder, input_image_idx, ROI_window, num_of_template, writeimagefolder):

    ## to run the WAMI tracker
    ## d_out  : output directory
    ## frames : a vector of frames to attack
    ref_track = None

    image_idx_offset = 0
    model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)

    # load transformation matrices
    matlabfile = hdf5storage.loadmat(model_folder+'Data/TransformationMatrices_train.mat')
    TransformationMatrices = matlabfile.get("TransMatrix")

    i0 = load_data[0]
    ROI_centre = load_data[1]
    bgt = load_data[2]
    kf = load_data[3]
    kf_attack = load_data[4]
    track_store = load_data[5]
    track_attack_store = load_data[6]

    for i in range(i0, 20):
        # Read input image
        frame_idx = input_image_idx + image_idx_offset + i
        ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
        input_image = ReadImage[ROI_centre[1] - ROI_window:ROI_centre[1] + ROI_window + 1,
                      ROI_centre[0] - ROI_window:ROI_centre[0] + ROI_window + 1]
        ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx - 1][0])
        ROI_centre = [int(i) for i in ROI_centre]

        Hs = bgt.doCalculateHomography(input_image)
        bgt.doMotionCompensation(Hs, input_image.shape)
        BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image,
                                                                                                    thres=8)

        dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres,
                                 BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression,
                                 aveImg_regression, attack)
        # dr.refinementID=refinementID
        refinedDetections, refinedProperties, ref = dr.doMovingVehicleRefinement()
        regressedDetections = dr.doMovingVehiclePositionRegression()
        regressedDetections = np.asarray(regressedDetections)

        # Kalman filter update
        if i0 > 0:
            # tracking without attack
            kf.TimePropagate(Hs[num_of_template - 1])
            kf.predict()
            kf.NearestNeighbourAssociator(regressedDetections)
            kf.update()
            track_x = kf.mu_t[0, 0]
            track_y = kf.mu_t[1, 0]
            # propagate all detections
            track_store = TimePropagate(track_store, Hs[num_of_template - 1])
            track_store.append(np.array([track_x, track_y]).reshape(2, 1))

            # tracking with attack
            kf_attack.TimePropagate(Hs[num_of_template - 1])
            kf_attack.predict()

            Neighbours = kf_attack.Neighbour_measurement(regressedDetections)

            regressionID = kf_attack.NearestNeighbourAssociator(regressedDetections)

            # the id in the refinement detections (input to the CNN)
            old_kfz = kf.z
            if isinstance(regressionID, np.int64):
                regression2refinedID = dr.regressedDetectionID[regressionID]
                refinementID = dr.refinedDetectionsID[regression2refinedID]

            else:
                refinementID = None


            kf_attack.update()
            track_attack_x = kf_attack.mu_t[0, 0]
            track_attack_y = kf_attack.mu_t[1, 0]
            # propagate all detections
            track_attack_store = TimePropagate(track_attack_store, Hs[num_of_template - 1])
            track_attack_store.append(np.array([track_attack_x, track_attack_y]).reshape(2, 1))

            # update background
            bgt.updateTemplate(input_image)

    res = diff_mean(track_store, track_attack_store)
    return res



