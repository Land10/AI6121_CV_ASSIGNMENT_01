import os
from GrayscaleHE import GrayscaleHE

os.chdir("./HE/AI6121_CV_Ass_01/")  # 替换为你想要切换到的路径!!!!!!!!!!!!
img_path = "input_images"
img_file = os.listdir(img_path)

for img in img_file :
    HE = GrayscaleHE(img_path, img)  # 建立实例对象
    HE.valification()  # 调用验证函数