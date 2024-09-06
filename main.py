import os
from GrayscaleHE import GrayscaleHE
<<<<<<< HEAD
from ColorfulPictureHE import ColorPictureHE
=======
from MultiChannelHE import RGBChannelHE, HSVChannelHE
>>>>>>> df0ec576b6eba185744db8af3c19ef23d30f6e57

if __name__ == '__main__':
    img_path = "input_images"
    img_file = os.listdir(img_path)

<<<<<<< HEAD
    #Processing Grayscale Historgram
    for img in img_file:
        HEG = GrayscaleHE(img_path,img)
        HEG.draw_histogram()
        HEG.equalization()
        HEG.draw_new_histogram()

    channels = ["YCrCb", "WeightedRGB", "RGB", "SharedRGB","HSV", "LAB", "HLS", "CALYCrCb","CALRGB"]
    for channel in channels:
        for img in img_file :
            HE = ColorPictureHE(img_path,img,channel)
            HE.draw_histogram()
            HE.equalization()
            HE.draw_new_histogram()
=======
    for img in img_file :
        # [GrayscaleHE, RGBChannelHE, HSVChannelHE]
        HE = HSVChannelHE(img_path, img)  # 建立实例对象
        HE.draw_histogram()  # 画原始图像直方图
        HE.equalization() # 利用以上计算得到的累积分布函数对图像进行均衡化，得到映射函数f,并使用映射函数f对原始图像进行均衡化，得到新的图像
        HE.draw_new_histogram() # 画新图像的直方图
>>>>>>> df0ec576b6eba185744db8af3c19ef23d30f6e57
