import os
from GrayscaleHE import GrayscaleHE
from RGBChannelHE import RGBChannelHE, RGBSharedChannelHE, RGBWeightedChannelHE
from HSVChannelHE import HSVChannelHE

if __name__ == '__main__':
    img_path = "input_images"
    img_file = os.listdir(img_path)

    for img in img_file :
        HE = HSVChannelHE(img_path, img)  # 建立实例对象
        HE.draw_histogram()  # 画原始图像直方图
        HE.equalization() # 利用以上计算得到的累积分布函数对图像进行均衡化，得到映射函数f,并使用映射函数f对原始图像进行均衡化，得到新的图像
        HE.draw_new_histogram() # 画新图像的直方图