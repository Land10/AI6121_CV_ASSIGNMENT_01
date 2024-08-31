import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class RGBChannelHE(object):
    def __init__(self, img_path, img_file):
        self.img_file = img_file
        self.OriginalImg = cv2.imread(os.path.join(img_path, img_file))
        self.Histogram = np.zeros((3, 256), dtype=np.intp)
        self.NewHistogram = np.zeros((3, 256), dtype=np.intp)
        self.Cumulative_distribution = np.zeros((3, 256), dtype=np.cfloat)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def draw_histogram(self):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        
        for row in range(height):
            for col in range(width):
                r = self.OriginalImg[row][col][0]
                self.Histogram[0][r] += 1
                g = self.OriginalImg[row][col][1]
                self.Histogram[1][g] += 1
                b = self.OriginalImg[row][col][2]
                self.Histogram[2][b] += 1
        
        plt.figure()
        plt.title('input image histogram')
        plt.subplot(3, 1, 1)
        plt.hist(self.OriginalImg[:,:,0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.OriginalImg[:,:,1].flatten(), bins=256, color='green')
        
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.OriginalImg[:,:,2].flatten(), bins=256, color='blue')
        
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", "RGBChannelHE", "origin_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "RGBChannelHE", "origin_histogram", self.img_file))

    def equalization(self):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width
        for i in range(256):
            if i == 0:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N
                self.Cumulative_distribution[1][i] = self.Histogram[1][i] / N
                self.Cumulative_distribution[2][i] = self.Histogram[2][i] / N
            elif i == 255:
                self.Cumulative_distribution[0][i] = 1.0
                self.Cumulative_distribution[1][i] = 1.0
                self.Cumulative_distribution[2][i] = 1.0
            else:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N + \
                                                     self.Cumulative_distribution[0][i - 1]
                self.Cumulative_distribution[1][i] = self.Histogram[1][i] / N + \
                                                     self.Cumulative_distribution[1][i - 1]
                self.Cumulative_distribution[2][i] = self.Histogram[2][i] / N + \
                                                     self.Cumulative_distribution[2][i - 1]

        self.Cumulative_distribution = self.Cumulative_distribution * 255
        # 绘制新图像
        self.NewImg = self.OriginalImg.copy()
        
        self.Cumulative_distribution = self.Cumulative_distribution.astype(np.intp)
        for row in range(height):
            for col in range(width):
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][0]]
                self.NewImg[row][col][0] = Newvalue
                Newvalue = self.Cumulative_distribution[1][self.NewImg[row][col][1]]
                self.NewImg[row][col][1] = Newvalue
                Newvalue = self.Cumulative_distribution[2][self.NewImg[row][col][2]]
                self.NewImg[row][col][2] = Newvalue
        os.makedirs(os.path.join("results", "RGBChannelHE", "result_images"), exist_ok=True)
        cv2.imwrite(os.path.join("results", "RGBChannelHE", "result_images", self.img_file), self.NewImg)

    def draw_new_histogram(self):
        plt.figure()
        plt.title('histogram after HE')
        plt.subplot(3, 1, 1)
        plt.hist(self.NewImg[:,:,0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.NewImg[:,:,1].flatten(), bins=256, color='green')
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.NewImg[:,:,2].flatten(), bins=256, color='blue')
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", "RGBChannelHE", "new_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "RGBChannelHE", "new_histogram", self.img_file))
        
        
class RGBSharedChannelHE(object):
    def __init__(self, img_path, img_file):
        self.img_file = img_file
        self.OriginalImg = cv2.imread(os.path.join(img_path, img_file))
        self.Histogram = np.zeros((1, 256), dtype=np.intp)
        self.NewHistogram = np.zeros((1, 256), dtype=np.intp)
        self.Cumulative_distribution = np.zeros((1, 256), dtype=np.cfloat)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def draw_histogram(self):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        
        for row in range(height):
            for col in range(width):
                r = self.OriginalImg[row][col][0]
                self.Histogram[0][r] += 1
                g = self.OriginalImg[row][col][1]
                self.Histogram[0][g] += 1
                b = self.OriginalImg[row][col][2]
                self.Histogram[0][b] += 1
        
        plt.figure()
        plt.title('input image histogram')
        plt.subplot(3, 1, 1)
        plt.hist(self.OriginalImg[:,:,0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.OriginalImg[:,:,1].flatten(), bins=256, color='green')
        
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.OriginalImg[:,:,2].flatten(), bins=256, color='blue')
        
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", "RGBSharedChannelHE", "origin_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "RGBSharedChannelHE", "origin_histogram", self.img_file))

    def equalization(self):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width
        for i in range(256):
            if i == 0:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N
            elif i == 255:
                self.Cumulative_distribution[0][i] = 1.0
            else:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N + \
                                                     self.Cumulative_distribution[0][i - 1]

        self.Cumulative_distribution = self.Cumulative_distribution * 255 / 3
        # 绘制新图像
        self.NewImg = self.OriginalImg.copy()
        
        self.Cumulative_distribution = self.Cumulative_distribution.astype(np.intp)
        for row in range(height):
            for col in range(width):
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][0]]
                self.NewImg[row][col][0] = Newvalue
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][1]]
                self.NewImg[row][col][1] = Newvalue
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][2]]
                self.NewImg[row][col][2] = Newvalue
        os.makedirs(os.path.join("results", "RGBSharedChannelHE", "result_images"), exist_ok=True)
        cv2.imwrite(os.path.join("results", "RGBSharedChannelHE", "result_images", self.img_file), self.NewImg)

    def draw_new_histogram(self):
        plt.figure()
        plt.title('histogram after HE')
        plt.subplot(3, 1, 1)
        plt.hist(self.NewImg[:,:,0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.NewImg[:,:,1].flatten(), bins=256, color='green')
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.NewImg[:,:,2].flatten(), bins=256, color='blue')
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", "RGBSharedChannelHE", "new_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "RGBSharedChannelHE", "new_histogram", self.img_file))
        
        
class RGBWeightedChannelHE(object):
    def __init__(self, img_path, img_file):
        self.img_file = img_file
        self.OriginalImg = cv2.imread(os.path.join(img_path, img_file))
        self.Histogram = np.zeros((1, 256), dtype=np.intp)
        self.NewHistogram = np.zeros((1, 256), dtype=np.intp)
        self.Cumulative_distribution = np.zeros((1, 256), dtype=np.cfloat)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def draw_histogram(self):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        GrayImg = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2GRAY)
        
        for row in range(height):
            for col in range(width):
                I = GrayImg[row][col]
                self.Histogram[0][I] += 1
        
        plt.figure()
        plt.title('input image histogram')
        plt.subplot(3, 1, 1)
        plt.hist(self.OriginalImg[:,:,0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.OriginalImg[:,:,1].flatten(), bins=256, color='green')
        
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.OriginalImg[:,:,2].flatten(), bins=256, color='blue')
        
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", "RGBWeightedChannelHE", "origin_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "RGBWeightedChannelHE", "origin_histogram", self.img_file))

    def equalization(self):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width
        for i in range(256):
            if i == 0:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N
            elif i == 255:
                self.Cumulative_distribution[0][i] = 1.0
            else:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N + \
                                                     self.Cumulative_distribution[0][i - 1]

        self.Cumulative_distribution = self.Cumulative_distribution * 255
        # 绘制新图像
        self.NewImg = self.OriginalImg.copy()
        
        self.Cumulative_distribution = self.Cumulative_distribution.astype(np.intp)
        for row in range(height):
            for col in range(width):
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][0]]
                self.NewImg[row][col][0] = Newvalue
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][1]]
                self.NewImg[row][col][1] = Newvalue
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][2]]
                self.NewImg[row][col][2] = Newvalue
        os.makedirs(os.path.join("results", "RGBWeightedChannelHE", "result_images"), exist_ok=True)
        cv2.imwrite(os.path.join("results", "RGBWeightedChannelHE", "result_images", self.img_file), self.NewImg)

    def draw_new_histogram(self):
        plt.figure()
        plt.title('histogram after HE')
        plt.subplot(3, 1, 1)
        plt.hist(self.NewImg[:,:,0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.NewImg[:,:,1].flatten(), bins=256, color='green')
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.NewImg[:,:,2].flatten(), bins=256, color='blue')
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", "RGBWeightedChannelHE", "new_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "RGBWeightedChannelHE", "new_histogram", self.img_file))
        