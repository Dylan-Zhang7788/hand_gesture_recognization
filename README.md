# hand_gesture_recognization


## 文件说明

手势识别方法，先使用PSP-Net对输入的图像进行分割，在此基础上使用VGG16进行分类。  
checkpoint文件中存储了PSP-Net和VGG16的权重。  
dataset文件用于存放训练使用的数据集，按照文件夹中的形式进行放置，由于的数据集过大，未进行无法上传。  
net文件中存储了各个网络。  
训练VGG网络时，直接运行train_VGG；训练PSP-net时，直接运行train_Seg。  
测试语义分割效果时，直接运行segmentation，该文件会将 图片1 进行语义分割，输出结果为mask.jpg  
实时进行手势识别，运行camera.py 使用普通的摄像头就可以。

## 环境说明

Ubuntu version: 20.04   
cuda version: 10.0  
cudnn version: 7.6.4  


matplotlib  
scipy==1.5.4  
numpy==1.17.0  
pandas==0.19.5  
seaborn  
scikit_learn  
theano==0.9.0  
h5py==2.10.0  
Pillow  
opencv-python  
torch==1.10.1  
torchvision==0.10.1  
torchaudio==0.11.2  
rospy  


**安装环境**
```
pip install -r requirements.txt
```
