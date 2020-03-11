import sys
sys.path.append('../DeepConcolic/src')
import numpy as np
import cv2
import timeit
import hdf5storage
import math
import TrainNetwork.TN_BaseFunctions as basefunctions
import matplotlib.pyplot as plt
from os import path
import pickle
from MovingObjectDetector.BackgroundModel import BackgroundModel
from MovingObjectDetector.DetectionRefinement import DetectionRefinement
from SimpleTracker.KalmanFilter import KalmanFilter
from MovingObjectDetector.MOD_BaseFunctions import TimePropagate, TimePropagate_, draw_error_ellipse2d
from MovingObjectDetector.Init_Track_From_Groundtruth import init_Track_From_Groundtruth


if __name__ == '__main__':
    # Parameters
    num_of_template = 3
    input_image_idx = 5
    imagefolder = 'E:/WPAFB-images/training/'
    model_folder = '../Models/'

    # load transformation matrices
    matlabfile = hdf5storage.loadmat('../Models/Data/TransformationMatrices_train.mat')
    TransformationMatrices = matlabfile.get("TransMatrix")

    # load CNN parameters
    model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)

    # Load background
    images = []
    Hs = []

    for i in range(num_of_template):
        frame_idx = input_image_idx + i - num_of_template
        ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
        images.append(ReadImage)
        H_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for j in range(num_of_template-i, 0, -1):
            frame_idx_for_H = input_image_idx - j
            H_ = TransformationMatrices[frame_idx_for_H - 1][0] @ H_
        Hs.append(H_)
    bgt = BackgroundModel(num_of_template=num_of_template, templates=images)
    bgt.Hs = Hs

    # moving object detection
    all_detections = {}
    all_id_conversion = {}
    frame_idx = input_image_idx
    frame_name = imagefolder + "frame%06d.png" % frame_idx
    while path.exists(frame_name):
        input_image = cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)
        bgt.doMotionCompensation(bgt.Hs, input_image.shape)
        # bgt.showCompensatedImages()
        # plt.figure()
        # plt.imshow(input_image)
        BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image, thres=8)

        dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres,
                                 BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression,
                                 aveImg_regression, attack=[])
        refinedDetections, refinedProperties, ref = dr.doMovingVehicleRefinement()
        regressedDetections = dr.doMovingVehiclePositionRegression()
        regressedDetections = np.asarray(regressedDetections)

        ID_regression2refinement = dr.ID_regression2refinement
        ID_refinement2background = dr.ID_refinement2background

        print('file name: ' + frame_name + ' --- number of detections: ' + str(len(regressedDetections)))

        with open('../DetectionsRecords/' + "frame%06d" % frame_idx + '_detections.bin', 'wb') as fid:
            pickle.dump((refinedDetections, regressedDetections), fid)
        with open('../DetectionsRecords/' + "frame%06d" % frame_idx + '_id_conversion.bin', 'wb') as fid:
            pickle.dump((ID_regression2refinement, ID_refinement2background), fid)

        # update background
        H_ = TransformationMatrices[frame_idx - 1][0]
        bgt.updateTemplate(input_image, H_)
        frame_idx += 1
        frame_name = imagefolder + "frame%06d.png" % frame_idx


