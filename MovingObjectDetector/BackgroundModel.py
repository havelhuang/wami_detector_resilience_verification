import cv2
import numpy as np
import MovingObjectDetector.ImageProcFunc as ImageProcessing
import matplotlib.pyplot as plt
import skimage.measure as measure
import skimage.filters as filters


class BackgroundModel:

    def __init__(self, num_of_template, templates):
        self.num_of_templates = num_of_template
        self.templates = templates
        self.motion_matrices = np.ndarray(shape=[3, 3, self.num_of_templates], dtype=np.float32)
        self.CompensatedImages = []
        self.background = []
        self.invalidArea = []
        self.Hs = None
        self.Background = []

    def showTemplate(self):
        for i in range(self.num_of_templates):
            plt.figure(i)
            plt.imshow(np.repeat(np.expand_dims(self.templates[i], -1), 3, axis=2))
            plt.show()

    def showCompensatedImages(self):
        for idx, cimg in enumerate(self.CompensatedImages):
            plt.figure(idx)
            plt.imshow(np.repeat(np.expand_dims(cimg, -1), 3, axis=2))
            plt.show()

    def getTemplates(self):
        return self.templates

    def getCompensatedImages(self):
        return self.CompensatedImages

    def updateTemplate(self, new_image, H_=None):
        num_of_templates = self.num_of_templates
        self.templates[0:num_of_templates-1] = self.templates[1:num_of_templates]
        self.templates[num_of_templates-1] = new_image
        self.Hs[0:num_of_templates-1] = self.Hs[1:num_of_templates]
        if H_ is None:
            self.Hs[num_of_templates-1] = []
        else:
            self.Hs[num_of_templates - 1] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            for i, m in enumerate(self.Hs):
                self.Hs[i] = H_ @ m
        return

    def doBackgroundSubtraction(self, input_image, thres=10):
        for i, thisBackground in enumerate(self.CompensatedImages):
            diff = np.float64(input_image) - np.float64(thisBackground)
            diff = cv2.GaussianBlur(diff, (21, 21), sigmaX=8)
            self.CompensatedImages[i] = np.uint8(thisBackground + diff)
        thisBackground = np.median(self.CompensatedImages, axis=0)
        self.Background = thisBackground
        subtractionResult = np.abs(input_image - thisBackground)
        subtractionResultBW = np.uint8(subtractionResult >= thres)
        subtractionResultBW[self.invalidArea] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        subtractionResultBW = cv2.morphologyEx(subtractionResultBW, cv2.MORPH_OPEN, kernel)
        labels = measure.label(subtractionResultBW, connectivity=1)
        Properties = measure.regionprops(labels)
        centres = []
        for thisProperty in Properties:
            centres.append([thisProperty.centroid[1], thisProperty.centroid[0]])
        centres = np.round(np.asarray(centres))
        return centres, Properties

    def doMotionCompensation(self, motion_matrix, dstShape):
        self.CompensatedImages = []
        validAreaAll = np.ones(dstShape, dtype=bool)
        for idx, srcImage in enumerate(self.templates):
            thisValidArea = np.ones(dstShape, dtype=bool)
            CompensatedImage = ImageProcessing.ImageRegistration(srcImage, dstShape, motion_matrix[idx])
            CalcValidArea = CompensatedImage == 255
            CalcValidArea = measure.label(CalcValidArea, connectivity=2)
            CalcValidAreaProp = measure.regionprops(CalcValidArea)
            maxArea = 0
            Coords = []
            for thisProperty in CalcValidAreaProp:
                if thisProperty.area > maxArea and thisProperty.area > 10000:
                    maxArea = thisProperty.area
                    Coords = thisProperty.coords
            if len(Coords) > 0:
                thisValidArea[Coords[:, 0], Coords[:, 1]] = False
                validAreaAll = validAreaAll & thisValidArea
            self.CompensatedImages.append(CompensatedImage)
        self.invalidArea = np.logical_not(validAreaAll)

    def doCalculateHomography(self, dst_image):
        Hs = []
        if not self.Hs:
            for srcImage in self.templates:
                H, _ = ImageProcessing.CalcHomography(srcImage, dst_image, num_of_features=2000)
                Hs.append(H)
        else:
            H, _ = ImageProcessing.CalcHomography(self.templates[self.num_of_templates-1], dst_image, num_of_features=2000)
            for idx in range(self.num_of_templates-1):
                Hs.append(np.matmul(H, self.Hs[idx]))
            Hs.append(H)
        self.Hs = Hs
        return Hs

