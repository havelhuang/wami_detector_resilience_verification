import numpy as np
import cv2
import timeit
import hdf5storage
import math
import TrainNetwork.TN_BaseFunctions as basefunctions
from copy import copy
import scipy.io as sio
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


def run_detection_main_greedy(attack, model_folder, imagefolder, input_image_idx, ROI_centre,
                       writeimagefolder, ROI_window,num_of_template):

    ## to run the WAMI tracker
    ## d_out  : output directory
    ## frames : a vector of frames to attack
    ref_track = None

    image_idx_offset = 0
    model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)

    # load transformation matrices
    matlabfile = hdf5storage.loadmat(model_folder+'Data/TransformationMatrices_train.mat')
    TransformationMatrices = matlabfile.get("TransMatrix")

    # Load background
    images = []
    for i in range(num_of_template):
        frame_idx = input_image_idx + image_idx_offset + i - num_of_template
        ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
        ReadImage = ReadImage[ROI_centre[1] - ROI_window:ROI_centre[1] + ROI_window + 1,
                    ROI_centre[0] - ROI_window:ROI_centre[0] + ROI_window + 1]
        ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx - 1][0])
        ROI_centre = [int(i) for i in ROI_centre]
        images.append(ReadImage)
    bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

    # Work out initialisation of a track with groundtruth
    frame_idx = input_image_idx + image_idx_offset
    min_r = ROI_centre[1] - ROI_window
    max_r = ROI_centre[1] + ROI_window
    min_c = ROI_centre[0] - ROI_window
    max_c = ROI_centre[0] + ROI_window
    show_available_tracks = True
    if show_available_tracks:
        ImageForInitTrack = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
        Init_Candidate_tracks = init_Track_From_Groundtruth(TransformationMatrices, frame_idx, (min_r, max_r, min_c, max_c), Image=ImageForInitTrack)
        print(Init_Candidate_tracks)
    else:
        Init_Candidate_tracks = init_Track_From_Groundtruth(TransformationMatrices, frame_idx, (min_r, max_r, min_c, max_c))

    # initialise Kalman filter
    init_idx = 53
    if init_idx >= len(Init_Candidate_tracks):
        init_idx = 0
        print("warning: the init track index is unavailable.")
    x = Init_Candidate_tracks[init_idx][0]
    y = Init_Candidate_tracks[init_idx][1]
    vx = Init_Candidate_tracks[init_idx][2]
    vy = Init_Candidate_tracks[init_idx][3]
    kf = KalmanFilter(np.array([[x], [y], [vx], [vy]]), np.diag([900, 900, 400, 400]), 5, 6)
    kf_attack = KalmanFilter_attack(np.array([[x], [y], [vx], [vy]]), np.diag([900, 900, 400, 400]), 5, 6)
    track_attack_store = []
    track_store = []
    perturbations = []

    starttime = timeit.default_timer()
    for i in range(20):
        print("Starting the step %s:"%i)
        # Read input image
        frame_idx = input_image_idx + image_idx_offset + i
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

        refinedDetections, refinedProperties,ref = dr.doMovingVehicleRefinement()
        regressedDetections = dr.doMovingVehiclePositionRegression()
        regressedDetections = np.asarray(regressedDetections)

        # Kalman filter update
        if i > 0:
            # tracking without attack
            kf.TimePropagate(Hs[num_of_template - 1])
            kf.predict()
            a = kf.NearestNeighbourAssociator(regressedDetections)
            print("Correct RegressionID is ",a)
            kf.update()
            track_x = kf.mu_t[0, 0]
            track_y = kf.mu_t[1, 0]
            # propagate all detections
            track_store = TimePropagate(track_store, Hs[num_of_template - 1])
            track_store.append(np.array([track_x, track_y]).reshape(2, 1))

            # tracking with attack
            kf_attack.TimePropagate(Hs[num_of_template - 1])
            kf_attack.predict()
            # the id in the regressed detections
            regressionID = kf_attack.NearestNeighbourAssociator(regressedDetections)
            print("Biased RegressionID is", regressionID)


            # the id in the refinement detections (input to the CNN)
            old_kfz = kf.z
            if isinstance(regressionID, np.int64):
                regression2refinedID = dr.regressedDetectionID[regressionID]
                refinementID = dr.refinedDetectionsID[regression2refinedID]
                print("Background subtraction id:" + str(refinementID))
                print("Background subtraction id type:" + str(type(refinementID)))
            else:
                refinementID = None
                print("Data Association failed (No detection is assigned to this track)...")

            # here to play 'attack': to call the dr again with refinementID
            if (i > 5) and (i < 10) :
                # return the nearest points
                Neighbours = kf_attack.NeighbourAssociation(regressedDetections,old_kfz)
                a = Neighbours[len(Neighbours) - 1]
                print("Biased RegressionID is", a)

                # # ideal case, the most distant point is associated
                # regressionID = Neighbours[len(Neighbours)-1]
                # kf_attack.z = regressedDetections[regressionID].reshape(2, 1)

                # attack the detected points in search range
                if len(Neighbours) != 1:
                    regressionID = Neighbours[len(Neighbours)-1]
                    Neighbours = Neighbours[:len(Neighbours)-1]
                    refinementID_set = []

                    regression2refinedID = dr.regressedDetectionID[regressionID]
                    refinementID_final = dr.refinedDetectionsID[regression2refinedID]


                    for regressionID in Neighbours :
                        regression2refinedID = dr.regressedDetectionID[regressionID]
                        refinementID = dr.refinedDetectionsID[regression2refinedID]

                        if isinstance(refinementID, np.int64):
                            refinementID_set.extend([refinementID])
                        else:
                            refinementID_set.extend(refinementID)

                    if isinstance(refinementID_final, np.int64):
                        if refinementID_final in refinementID_set:
                            refinementID_set.remove(refinementID_final)
                    else:
                        for refinementID in refinementID_final:
                            if refinementID in refinementID_set:
                                refinementID_set.remove(refinementID)

                    dr.refinementID = refinementID_set
                    refinedDetections, refinedProperties, ref = dr.doMovingVehicleRefinement()
                    regressedDetections = dr.doMovingVehiclePositionRegression()
                    regressedDetections = np.asarray(regressedDetections)
                    perturbations.extend(dr.perturbations)

                regressionID = kf_attack.NearestNeighbourAssociator(regressedDetections)



                new_kfz = kf_attack.z
                print('*********************')
                print(old_kfz)
                print('####')
                print(new_kfz)
                print('*********************')
                # the id in the refinement detections (input to the CNN)
                print('#### old refinementID', refinementID)
                if regressionID is None:
                    print('#### new refinementID does not exist, because there is no associated detection')
                else:
                    regression2refinedID = dr.regressedDetectionID[regressionID]
                    refinementID = dr.refinedDetectionsID[regression2refinedID]
                    print('#### new refinementID', refinementID)


            kf_attack.update()
            track_attack_x = kf_attack.mu_t[0, 0]
            track_attack_y = kf_attack.mu_t[1, 0]
            # propagate all detections
            track_attack_store = TimePropagate(track_attack_store, Hs[num_of_template - 1])
            track_attack_store.append(np.array([track_attack_x, track_attack_y]).reshape(2, 1))
            print('Estimated State (Attacked): ' + str(kf.mu_t.transpose()))
        else:
            track_attack_x = kf_attack.mu_t[0, 0]
            track_attack_y = kf_attack.mu_t[1, 0]
            track_attack_store.append(np.array([track_attack_x, track_attack_y]).reshape(2, 1))
            track_x = kf.mu_t[0, 0]
            track_y = kf.mu_t[1, 0]
            track_store.append(np.array([track_x, track_y]).reshape(2, 1))

        # update background
        bgt.updateTemplate(input_image)

        # plt.figure()
        minx = np.int32(track_attack_x - 300)
        if minx <= 0:
            minx = 1
        miny = np.int32(track_attack_y - 300)
        if miny <= 0:
            miny = 1
        maxx = np.int32(track_attack_x + 301)
        if maxx >= input_image.shape[1]:
            maxx = input_image.shape[1]
        maxy = np.int32(track_attack_y + 301)
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

        for idx in range(1, len(track_attack_store)):
            point1x = np.int32(track_attack_store[idx - 1][0, 0]) - minx
            point1y = np.int32(track_attack_store[idx - 1][1, 0]) - miny
            point2x = np.int32(track_attack_store[idx][0, 0]) - minx
            point2y = np.int32(track_attack_store[idx][1, 0]) - miny
            cv2.line(roi_image, (point1x, point1y), (point2x, point2y), (0, 0, 255), 2)
        for idx in range(1, len(track_store)):
            point1x = np.int32(track_store[idx - 1][0, 0]) - minx
            point1y = np.int32(track_store[idx - 1][1, 0]) - miny
            point2x = np.int32(track_store[idx][0, 0]) - minx
            point2y = np.int32(track_store[idx][1, 0]) - miny
            cv2.line(roi_image, (point1x, point1y), (point2x, point2y), (0, 255, 0), 2)

        # draw_error_ellipse2d(roi_image, (kf1.mu_t[0]-minx, kf1.mu_t[1]-miny), kf1.sigma_t)
        # cv2.circle(input_image, (np.int32(trackx), np.int32(tracky)), 15, (255, 0, 0), 3)
        #print("writing into %s"%(writeimagefolder + "%05d.png" % i))
        cv2.imwrite(writeimagefolder + "%05d.png" % i, roi_image)
        """
        plt.figure()
        plt.imshow(np.repeat(np.expand_dims(input_image, -1), 3, axis=2))
        #plt.plot(BackgroundSubtractionCentres[:,0], BackgroundSubtractionCentres[:,1], 'g.')
        #plt.plot(refinedDetections[:,0], refinedDetections[:,1], 'y.')
        plt.plot(np.int32(regressedDetections[:,0]), np.int32(regressedDetections[:,1]), 'r.', markersize=3)
        plt.plot(np.int32(trackx), np.int32(tracky), 'yo', markersize=5)
        plt.show()
        """
        print('-------------------------------------------------------------------------------------------------------')
        res = diff_mean(track_store, track_attack_store)
        print("the averaged deviation is ", res)
        print('-------------------------------------------------------------------------------------------------------')


    print("the average perturbation added in the attack is " , (sum(perturbations)/len(perturbations)))
    print("the total perturbation added in the attack is " , (sum(perturbations)))

    endtime = timeit.default_timer()
    print("Processing Time (Total): " + str(endtime - starttime) + " s... ")
    sio.savemat('greedy.mat', {'greedy': kf_attack.uncertainty})

    plt.figure()
    x = range(1,20)
    plt.plot(x, kf_attack.uncertainty)
    plt.xlim((1, 20))
    my_x_ticks = np.arange(1, 20, 1)
    plt.xticks(my_x_ticks,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time Step',fontsize=30)
    plt.ylabel('Uncertainty',fontsize=30)
    plt.show()

    # for i in range(1,21):
    #     plt.figure()
    #     x = range(1,i)
    #     plt.plot(x, kf_attack.uncertainty[:i-1])
    #     plt.xlim((1, 20))
    #     my_x_ticks = np.arange(1, 20, 1)
    #     plt.xticks(my_x_ticks, fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.xlabel('Time Step', fontsize=15)
    #     plt.ylabel('Uncertainty', fontsize=15)
    #     plt.savefig('%05d.png' % i)