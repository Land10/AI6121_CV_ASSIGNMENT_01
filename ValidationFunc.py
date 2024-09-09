import os
from ColorfulPictureHE import ColorPictureHE

os.chdir("./HE/AI6121_CV_Ass_01/")  # 替换为你的路径
print(os.getcwd())  # 打印当前工作目录
img_path = "input_images"
img_file = os.listdir(img_path)

for img in img_file :
    HE = ColorPictureHE(img_path, img)  # 建立实例对象
    HE.valification()  # 调用验证函数