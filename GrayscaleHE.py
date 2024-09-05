import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class GrayscaleHE(object):
    def __init__(self, img_path, img_file):
        self.img_file = img_file
        self.OriginalImg = cv2.imread(os.path.join(img_path, img_file))
        self.Histogram = np.zeros((1, 256), dtype=np.intp)
        self.NewHistogram = np.zeros((1, 256), dtype=np.intp)
        self.Cumulative_distribution = np.zeros((1, 256), dtype=np.float64)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

    def resize(self, size: tuple = (64, 64)):
        self.OriginalImg = cv2.resize(self.OriginalImg, size)

    def draw_histogram(self):
        """
        转为灰度图，统计像素灰度数据，绘制直方图
        :return: None
        """
        # 先转化为灰度图
        self.OriginalImg = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('./result/origin_img.jpg', self.OriginalImg)
        # 统计各种像素点的数量，存入Histogram
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        for row in range(height):
            for col in range(width):
                I = self.OriginalImg[row][col]
                self.Histogram[0][I] += 1
        # 绘制直方图
        x = np.asarray(self.OriginalImg)
        x.resize((height * width))
        plt.figure()
        plt.hist(x, bins=256, color='green')
        plt.xlabel('gray level')
        plt.ylabel('frequency')
        plt.title('input image histogram')
        os.makedirs(os.path.join("results", "GrayscaleHE", "origin_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "GrayscaleHE", "origin_histogram", self.img_file))
        x.resize((height, width))
        plt.close()

    def equalization(self):
        """
        利用计算得到的累积分布函数对原始图像像素进行均衡化，得到映射函数
        :return: None
        """
        # 累积分布函数计算完成后，进行I和c的缩放，把值域缩放到0~255的范围之内
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width # 所有像素点个数
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
                Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col]]
                self.NewImg[row][col] = Newvalue
        os.makedirs(os.path.join("results", "GrayscaleHE", "result_images"), exist_ok=True)
        cv2.imwrite(os.path.join("results", "GrayscaleHE", "result_images", self.img_file), self.NewImg)

    def draw_new_histogram(self):
        """绘制新图片的直方图"""
        # 计算像素点，得到原始图片直方图
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        for row in range(height):
            for col in range(width):
                I = self.NewImg[row][col]
                self.NewHistogram[0][I] += 1
        # 绘制直方图
        self.NewImg.resize((height*width))
        plt.figure()
        plt.hist(self.NewImg, bins=256, color='green')
        plt.xlabel('gray level')
        plt.ylabel('frequency')
        plt.title('histogram after HE')
        os.makedirs(os.path.join("results", "GrayscaleHE", "new_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", "GrayscaleHE", "new_histogram", self.img_file))
        self.NewImg.resize((height, width))
        plt.close()
        
