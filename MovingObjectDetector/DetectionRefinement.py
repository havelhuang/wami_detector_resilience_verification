import numpy as np
import sys
import TrainNetwork.TN_BaseFunctions as bf
from sklearn.neighbors import NearestNeighbors
import skimage.measure as measure
import cv2
from art.attacks.evasion.fast_gradient import FastGradientMethod
from DeepConcolic.src.mcdc import mcdc
from art.attacks.evasion.deepfool import DeepFool
from art.classifiers.keras import KerasClassifier


class DetectionRefinement:

    def __init__(self, input_image, compensatedImages, BackgroundSubtractionDetections, BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression, aveImg_regression, attack):
        self.num_of_template = len(compensatedImages)
        self.img_t = input_image
        self.img_tminus1 = compensatedImages[self.num_of_template-1]
        self.img_tminus2 = compensatedImages[self.num_of_template-2]
        self.img_tminus3 = compensatedImages[self.num_of_template-3]
        self.original_detections = BackgroundSubtractionDetections
        self.detections = BackgroundSubtractionDetections
        self.bgProperties = BackgroundSubtractionProperties
        self.model_binary = model_binary
        self.aveImg_binary = aveImg_binary
        self.model_regression = model_regression
        self.aveImg_regression = aveImg_regression
        self.refinedDetectionsID = []
        self.regressedDetectionID = []
        self.perturbations = []
        self.refinementID=None
        self.attack = attack

    def doMovingVehicleRefinement(self):
        # attack successfully ref = 1, otherwise ref = 0
        ref = 0
        img_shape = self.img_t.shape
        width = img_shape[1]
        height = img_shape[0]
        num_points = len(self.original_detections)
        X = np.ndarray((num_points, 4, 21, 21), dtype=np.float32)
        mask1 = np.zeros(num_points, dtype=np.bool)
        for i, thisdetection in enumerate(self.original_detections):
            minx = np.int32(np.round(thisdetection[0] - 10))
            miny = np.int32(np.round(thisdetection[1] - 10))
            maxx = np.int32(np.round(thisdetection[0] + 11))
            maxy = np.int32(np.round(thisdetection[1] + 11))
            if minx > 0 and miny > 0 and maxx < width and maxy < height:
                data_t = np.reshape(self.img_t[miny:maxy, minx:maxx], (1, 21, 21))
                data_tminus1 = np.reshape(self.img_tminus1[miny:maxy, minx:maxx], (1, 21, 21))
                data_tminus2 = np.reshape(self.img_tminus2[miny:maxy, minx:maxx], (1, 21, 21))
                data_tminus3 = np.reshape(self.img_tminus3[miny:maxy, minx:maxx], (1, 21, 21))
                X[i] = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
                mask1[i] = True
        X = X[mask1, ...]
        detections = self.original_detections[mask1, ...]
        X, _ = bf.DataNormalisationZeroCentred(X, self.aveImg_binary)

        # adversarial attack on input image
        if ("classification" in self.attack) and not (self.refinementID is None):

            x_test = X[self.refinementID]
            classifier = KerasClassifier(self.model_binary)
            pred1 = np.argmax(classifier.predict(x_test), axis=1)

            # # Craft adversarial samples with FGSM
            # epsilon = 50
            # adv_crafter = BasicIterativeMethod(classifier,eps= epsilon , eps_step=5, max_iter=1000)
            # x_test_adv = adv_crafter.generate(x = x_test)
            # X[self.refinementID] = x_test_adv

            # # Craft adversarial samples with DeepConcolic
            # res, testSuite = mcdc(X[self.refinementID], self.model_binary, self.aveImg_binary, mcdc_cond_ratio=0.99)
            # X[self.refinementID] = testSuite

            # Craft adversarial samples with DeepFool
            adv_crafter = DeepFool(classifier)
            x_test_adv = adv_crafter.generate(x=x_test,nb_grads=2)
            X[self.refinementID] = x_test_adv
            pred2 = np.argmax(classifier.predict(x_test_adv), axis=1)

            # calculate the difference between two sequence of images
            for i in range(x_test.shape[0]):
                for j in range(x_test.shape[1]):
                    x1 = x_test[i][j].flatten('F')
                    x2 = x_test_adv[i][j].flatten('F')
                    diff = np.sqrt(sum(np.square(x1 - x2))) / (x_test.shape[2]*x_test.shape[3])
                    self.perturbations.append(diff)


            if (pred1 == pred2).all():
                ref = 1

            self.refinementID = None

        predictResults = self.model_binary.predict(X, batch_size=1000, verbose=0)

        mask2 = np.zeros(len(predictResults), dtype=np.bool)
        for idx in range(len(predictResults)):
            thisResult = predictResults[idx]
            if thisResult[0] > 0.5:
                mask2[idx] = True
        refinedDetections = detections[mask2, ...]
        refinedProperties = []
        self.refinedDetectionsID = np.where(mask2)[0]
        mask3 = mask1
        mask3[mask1] = mask2
        for i, thisProperty in enumerate(self.bgProperties):
            if mask3[i] is True:
                refinedProperties.append(thisProperty)
        self.detections = refinedDetections
        self.bgProperties = refinedProperties
        return refinedDetections, refinedProperties,ref

    def doMovingVehiclePositionRegression(self):
        bgProperties = self.bgProperties
        img_shape = self.img_t.shape
        width = img_shape[1]
        height = img_shape[0]
        regressedDetections = []
        regressedDetectionsID = []
        processedDetections = np.zeros((len(self.detections)), dtype=np.bool)
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(self.detections)
        for i, thisdetection in enumerate(self.detections):
            if not processedDetections[i]:
                distances, indices = nbrs.kneighbors(thisdetection.reshape(1, -1))
                distances = distances.reshape(distances.size)
                indices = indices.reshape(indices.size)
                neiIdx = indices[(distances <= 15) & (~processedDetections[indices])]
                processedDetections[neiIdx] = True
                if len(neiIdx) == 1:
                    regressedDetections.append([thisdetection[0], thisdetection[1]])
                    if neiIdx != i:
                        print("The nearest point should be itself...")
                    regressedDetectionsID.append(i)
                elif len(neiIdx) > 1:
                    neiCentres = self.detections[neiIdx]
                    MeanCentre = np.mean(neiCentres, axis=0)
                    minx = np.round(MeanCentre[0])-22
                    maxx = np.round(MeanCentre[0])+23
                    miny = np.round(MeanCentre[1])-22
                    maxy = np.round(MeanCentre[1])+23
                    if minx > 0 and miny > 0 and maxx < width and maxy < height:
                        minx = np.int32(minx)
                        maxx = np.int32(maxx)
                        miny = np.int32(miny)
                        maxy = np.int32(maxy)
                        data_t = np.reshape(self.img_t[miny:maxy, minx:maxx], (1, 45, 45))
                        data_tminus1 = np.reshape(self.img_tminus1[miny:maxy, minx:maxx], (1, 45, 45))
                        data_tminus2 = np.reshape(self.img_tminus2[miny:maxy, minx:maxx], (1, 45, 45))
                        data_tminus3 = np.reshape(self.img_tminus3[miny:maxy, minx:maxx], (1, 45, 45))
                        X = np.concatenate((data_t, data_tminus1, data_tminus2, data_tminus3), axis=0)
                        X = np.expand_dims(X, axis=0)
                        X, _ = bf.DataNormalisationZeroCentred(X, self.aveImg_regression)
                        if ("regression" in self.attack) and not (self.refinementID is None):
                            res, X[0] = mcdc_regression_linf(X[0], self.model_regression, self.aveImg_regression, regression_threshold = 0.1, mcdc_cond_ratio=0.99)
                        RegressionResult = self.model_regression.predict(X, batch_size=1, verbose=0)
                        RegressionResult = cv2.resize(np.reshape(RegressionResult, (15, 15)), (45, 45))
                        MaxRegressionValue = np.max(RegressionResult)
                        tmpxy = np.where(RegressionResult == MaxRegressionValue)
                        tmpDetections = []
                        tmpDetections.append([tmpxy[1][0]+minx, tmpxy[0][0]+miny])
                        regressedDetectionsID.append(neiIdx)
                        MaxRegressionValue -= 0.1
                        while MaxRegressionValue >= 0.25:
                            RegressionResult_bw = RegressionResult >= MaxRegressionValue
                            labels = measure.label(RegressionResult_bw, connectivity=1)
                            maxLabel = np.max(labels)
                            for ii in range(1, maxLabel+1):
                                validRegressionResult = np.zeros(RegressionResult.shape)
                                validRegressionResult[labels == ii] = RegressionResult[labels == ii]
                                [tmpy, tmpx] = np.unravel_index(np.argmax(validRegressionResult, axis=None), RegressionResult.shape)
                                tmpx = tmpx+minx
                                tmpy = tmpy+miny
                                if not ismember([tmpx, tmpy], tmpDetections):
                                    tmpDetections.append([tmpx, tmpy])
                                    regressedDetectionsID.append(neiIdx)
                            MaxRegressionValue -= 0.1
                        regressedDetections.extend(tmpDetections)
        self.regressedDetectionID = regressedDetectionsID
        return regressedDetections


def ismember(A, B):
    output = False
    for b in B:
        if A == b:
            output = True
            break
    return output

