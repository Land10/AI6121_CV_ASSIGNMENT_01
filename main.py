import os
from GrayscaleHE import GrayscaleHE
from ColorfulPictureHE import ColorPictureHE

if __name__ == '__main__':
    img_path = "input_images"
    img_file = os.listdir(img_path)

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