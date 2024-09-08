import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class ColorPictureHE(object):
    def __init__(self, img_path, img_file, channel = "WeightRGB"):
        self.img_file = img_file
        self.OriginalImg = cv2.imread(os.path.join(img_path, img_file))
        self.Histogram = np.zeros((3, 256), dtype=np.intp)
        self.NewHistogram = np.zeros((3, 256), dtype=np.intp)
        self.Cumulative_distribution = np.zeros((3, 256), dtype=np.float64)
        self.channel = channel
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
        plt.hist(self.OriginalImg[:, :, 0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.OriginalImg[:, :, 1].flatten(), bins=256, color='green')

        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.OriginalImg[:, :, 2].flatten(), bins=256, color='blue')

        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "origin_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", f"{self.channel}ChannelHE", "origin_histogram", self.img_file))
        plt.close()

    def equalization(self, CAL = False):
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width
        if self.channel == "WeightedRGB" or self.channel == "SharedRGB":
            pass
        else:
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
            self.NewImg = self.OriginalImg.copy()
    
            self.Cumulative_distribution = self.Cumulative_distribution.astype(np.intp)

        if self.channel == "RGB":
            for row in range(height):
                for col in range(width):
                        Newvalue = self.Cumulative_distribution[0][self.NewImg[row][col][0]]
                        self.NewImg[row][col][0] = Newvalue
                        Newvalue = self.Cumulative_distribution[1][self.NewImg[row][col][1]]
                        self.NewImg[row][col][1] = Newvalue
                        Newvalue = self.Cumulative_distribution[2][self.NewImg[row][col][2]]
                        self.NewImg[row][col][2] = Newvalue
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "WeightedRGB":
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
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "SharedRGB":
            for i in range(256):
                if i == 0:
                    self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N
                elif i == 255:
                    self.Cumulative_distribution[0][i] = 1.0
                else:
                    self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N + \
                                                         self.Cumulative_distribution[0][i - 1]

            self.Cumulative_distribution = self.Cumulative_distribution * 255 / 3
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
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "HSV":
            hsv_image = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            for row in range(height):
                for col in range(width):
                    v_new_value = self.Cumulative_distribution[2][v[row, col]]
                    v[row, col] = v_new_value
            hsv_image_eq = cv2.merge([h, s, v])
            self.NewImg = cv2.cvtColor(hsv_image_eq, cv2.COLOR_HSV2BGR)
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "YCrCb":
            ycrcb_image = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb_image)
            for row in range(height):
                for col in range(width):
                    y_new_value = self.Cumulative_distribution[0][y[row, col]]
                    y[row, col] = y_new_value
            ycrcb_image_eq = cv2.merge([y, cr, cb])
            self.NewImg = cv2.cvtColor(ycrcb_image_eq, cv2.COLOR_YCrCb2BGR)
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "LAB":
            lab_image = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab_image)
            for row in range(height):
                for col in range(width):
                    l_new_value = self.Cumulative_distribution[0][l[row, col]]
                    l[row, col] = l_new_value
            lab_image_eq = cv2.merge([l, a, b])
            self.NewImg = cv2.cvtColor(lab_image_eq, cv2.COLOR_Lab2BGR)
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "HLS":
            hls_image = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2HLS)
            h, l, s = cv2.split(hls_image)
            for row in range(height):
                for col in range(width):
                    l_new_value = self.Cumulative_distribution[1][l[row, col]]
                    l[row, col] = l_new_value
            hls_image_eq = cv2.merge([h, l, s])
            self.NewImg = cv2.cvtColor(hls_image_eq, cv2.COLOR_HLS2BGR)
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "CALYCrCb":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            ycrcb_image = cv2.cvtColor(self.OriginalImg, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb_image)
            y = clahe.apply(y)
            ycrcb_image_eq = cv2.merge([y, cr, cb])
            self.NewImg = cv2.cvtColor(ycrcb_image_eq, cv2.COLOR_YCrCb2BGR)
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        elif self.channel == "CALRGB":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            for i in range(3):
                self.NewImg[:, :, i] = clahe.apply(self.OriginalImg[:, :, i])
            os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "result_images"), exist_ok=True)
            cv2.imwrite(os.path.join("results", f"{self.channel}ChannelHE", "result_images", self.img_file), self.NewImg)

        else:
            raise ValueError(f"Unknown Channel: {self.channel}. Available Channels are RGB, WeightedRGB, SharedRGB, HSV, YCrCb, LAB, HLS, CALHE.")

    def draw_new_histogram(self):
        plt.figure()
        plt.title('histogram after HE')
        plt.subplot(3, 1, 1)
        plt.hist(self.NewImg[:, :, 0].flatten(), bins=256, color='red')
        plt.subplot(3, 1, 2)
        plt.hist(self.NewImg[:, :, 1].flatten(), bins=256, color='green')
        plt.ylabel('frequency')
        plt.subplot(3, 1, 3)
        plt.hist(self.NewImg[:, :, 2].flatten(), bins=256, color='blue')
        plt.xlabel('gray level')
        os.makedirs(os.path.join("results", f"{self.channel}ChannelHE", "new_histogram"), exist_ok=True)
        plt.savefig(os.path.join("results", f"{self.channel}ChannelHE", "new_histogram", self.img_file))
        plt.close()

