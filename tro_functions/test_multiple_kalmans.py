import sys
sys.path.append('../DeepConcolic/src')
import glob
import re
import pickle
from copy import copy
from copy import deepcopy
import matplotlib.pyplot as plt
import hdf5storage
import cv2
import numpy as np
import timeit
import TrainNetwork.TN_BaseFunctions as basefunctions
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
from SimpleTracker.KalmanFilter import KalmanFilter
from SimpleTracker.MultiKalmanFilters import MultiKalmanFilters
from MovingObjectDetector.MOD_BaseFunctions import TimePropagate, TimePropagate_, draw_error_ellipse2d
from MovingObjectDetector.Init_Track_From_Groundtruth import init_Track_From_Groundtruth


# def track_multikf_main_baseline(attack, model_folder, imagefolder, detectionfolder, input_image_idx, ROI_centre,
#                        writeimagefolder, ROI_window, num_of_template):

# for testing
imagefolder = 'E:/WPAFB-images/training/'
model_folder = '../Models/'
num_of_template = 3
input_image_idx = 5
ROI_centre = [4500, 4500]
ROI_window = 1000
writeimagefolder = '../savefig/'
attack = []

model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)

# load transformation matrices
matlabfile = hdf5storage.loadmat(model_folder + 'Data/TransformationMatrices_train.mat')
TransformationMatrices = matlabfile.get("TransMatrix")

# Load background
images = []
for i in range(num_of_template):
    frame_idx = input_image_idx + i - num_of_template
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    ReadImage = ReadImage[ROI_centre[1] - ROI_window:ROI_centre[1] + ROI_window + 1,
                ROI_centre[0] - ROI_window:ROI_centre[0] + ROI_window + 1]
    ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx - 1][0])
    ROI_centre = [int(i) for i in ROI_centre]
    images.append(ReadImage)
bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

# Work out initialisation of a track with groundtruth
frame_idx = input_image_idx
min_r = ROI_centre[1] - ROI_window
max_r = ROI_centre[1] + ROI_window
min_c = ROI_centre[0] - ROI_window
max_c = ROI_centre[0] + ROI_window
show_available_tracks = True
if show_available_tracks:
    ImageForInitTrack = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    Init_Candidate_tracks = init_Track_From_Groundtruth(TransformationMatrices, frame_idx, (min_r, max_r, min_c, max_c),
                                                        groundtruth_path='../Models/ground-truth.csv', Image=ImageForInitTrack)
    print(Init_Candidate_tracks)
else:
    Init_Candidate_tracks = init_Track_From_Groundtruth(TransformationMatrices, frame_idx, (min_r, max_r, min_c, max_c))

# initialise Kalman filter
init_idx = 39
if init_idx >= len(Init_Candidate_tracks):
    init_idx = 0
    print("warning: the init track index is unavailable.")
x = Init_Candidate_tracks[init_idx][0]
y = Init_Candidate_tracks[init_idx][1]
vx = Init_Candidate_tracks[init_idx][2]
vy = Init_Candidate_tracks[init_idx][3]
kf = KalmanFilter(np.array([[x], [y], [vx], [vy]]), np.diag([15**2, 15**2, 10**2, 10**2]), 5, 6)
mkf = MultiKalmanFilters()

# start iterations
# kf_attack = deepcopy(kf)
track_attack_store = []
track_store = []
perturbations = []
starttime = timeit.default_timer()
for i in range(1, 60):
    print("Starting the step %s:" % i)

    # Read input image
    frame_idx = input_image_idx + i
    ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
    input_image = ReadImage[ROI_centre[1] - ROI_window:ROI_centre[1] + ROI_window + 1,
                  ROI_centre[0] - ROI_window:ROI_centre[0] + ROI_window + 1]
    ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx - 1][0])
    ROI_centre = [int(i) for i in ROI_centre]

    Hs = bgt.doCalculateHomography(input_image)
    bgt.doMotionCompensation(Hs, input_image.shape)
    BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image, thres=8)

    dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres,
                             BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression,
                             aveImg_regression, attack)
    # dr.refinementID=refinementID
    refinedDetections, refinedProperties, ref = dr.doMovingVehicleRefinement()
    regressedDetections = dr.doMovingVehiclePositionRegression()
    regressedDetections = np.asarray(regressedDetections)


    # Kalman filter update
    # tracking without attack
    kf.TimePropagate(Hs[num_of_template - 1])
    mkf.TimePropagate(Hs[num_of_template - 1])

    kf.predict()
    mkf.predict()

    # kf.NearestNeighbourAssociator(regressedDetections)
    primaryTrack, primary_masurement_ID = mkf.MaximumLikelihoodAssociator(kf, regressedDetections)
    kf = primaryTrack

    kf.update()
    mkf.update()

    mkf.InitMultiKalmanFilters(kf)

    track_x = kf.mu_t[0, 0]
    track_y = kf.mu_t[1, 0]
    # propagate all detections
    track_store = TimePropagate(track_store, Hs[num_of_template - 1])
    track_store.append(np.array([track_x, track_y]).reshape(2, 1))

    # update background
    bgt.updateTemplate(input_image)

    # plt.figure()
    minx = np.int32(track_x - 300)
    if minx <= 0:
        minx = 1
    miny = np.int32(track_y - 300)
    if miny <= 0:
        miny = 1
    maxx = np.int32(track_x + 301)
    if maxx >= input_image.shape[1]:
        maxx = input_image.shape[1]
    maxy = np.int32(track_y + 301)
    if maxy >= input_image.shape[0]:
        maxy = input_image.shape[0]
    print("write roi image windows: " + str(miny) + "," + str(maxy) + "," + str(minx) + "," + str(maxx))
    roi_image = np.repeat(np.expand_dims(input_image[miny:maxy, minx:maxx], -1), 3, axis=2)
    # Not necessary: cv2.circle(roi_image, (301, 301), 10, (255, 0, 0), 1)
    validRegressedDetections = np.int32(copy(regressedDetections))
    validRegressedDetections[:, 0] = validRegressedDetections[:, 0] - minx
    validRegressedDetections[:, 1] = validRegressedDetections[:, 1] - miny
    for thisDetection in validRegressedDetections:
        if (thisDetection[0] > 0) and (thisDetection[0] < 600) and (thisDetection[1] > 0) and (thisDetection[1] < 600):
            cv2.circle(roi_image, (thisDetection[0], thisDetection[1]), 3, (100, 100, 0), -1)
    for idx in range(1, len(track_store)):
        point1x = np.int32(track_store[idx - 1][0, 0]) - minx
        point1y = np.int32(track_store[idx - 1][1, 0]) - miny
        point2x = np.int32(track_store[idx][0, 0]) - minx
        point2y = np.int32(track_store[idx][1, 0]) - miny
        cv2.line(roi_image, (point1x, point1y), (point2x, point2y), (0, 255, 0), 2)
    cv2.imwrite(writeimagefolder + "%05d.png" % i, roi_image)

    print('-------------------------------------------------------------------------------------------------------')

endtime = timeit.default_timer()
print("Processing Time (Total): " + str(endtime - starttime) + " s... ")





'''def load_detections(detection_folder='DetectionsRecords/'):
    detection_filenames = glob.glob(detection_folder + '*_detections.bin')
    idconversion_filenames = glob.glob(detection_folder + '*_conversions.bin')

    detections_dict = {}
    for d_f in detection_filenames:
        re_search = re.search('frame([0-9]*)', d_f)
        with open(d_f, 'rb') as fid:
            inp = pickle.load(fid)
            detections_dict[re_search.group(1)] = copy(inp[1])
    idconversions_dict = {}
    for c_f in idconversion_filenames:
        re_search = re.search('frame([0-9]*)', c_f)
        with open(c_f, 'rb') as fid:
            inp = pickle.load(fid)
            idconversions_dict[re_search.group(1)] = copy(inp)

    return detections_dict, idconversions_dict


def detections_after_crop(detections, ROI_centre, ROI_window):
    min_r = ROI_centre[1] - ROI_window
    max_r = ROI_centre[1] + ROI_window
    min_c = ROI_centre[0] - ROI_window
    max_c = ROI_centre[0] + ROI_window
    new_detections = []
    for detection in detections:
        newX = detection[0] - min_c
        newY = detection[1] - min_r
        if (newX > 5) and (newY > 5) and (newX < ROI_window*2 - 5) and (newY < ROI_window*2 - 5):
            new_detections.append([newX, newY])

    return np.array(new_detections)'''

