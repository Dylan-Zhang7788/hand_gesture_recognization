#!/usr/bin/env python
import numpy as np
import cv2
import time
import os
import torch
import rospy
from predict import predict
from segmentation import segment, build_network
from std_msgs.msg import String

#init segement
snapshot = os.path.join('./checkpoints', 'densenet', 'PSPNet_last')
net, starting_epoch = build_network(snapshot, 'densenet')
net.eval()

# init VGG16
Myvggc2 = torch.load("./checkpoints/VGG16/VGG_trained.pkl")
Myvggc2 = Myvggc2.cuda()
Myvggc2.eval()

#init camera
video = cv2.VideoCapture(0)
index=0

pub = rospy.Publisher('result', String , queue_size=1)
rospy.init_node('result_pub', anonymous=True)

while video.isOpened():
    ret, frame = video.read()
    if ret is True:
        frame = cv2.flip(frame, 1)

    top, right, bottom, left = 10, 350, 234, 574
    roi = frame[top:bottom, right:left]
    #roi=cv2.flip(roi,1)
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow("Video Feed", frame)
    if index%30==0:
        blank=np.ones((70,350,3))
        hand=segment(net,roi)
        n=torch.sum(hand[0,:,112])/112
        if(n<50):
            cv2.putText(blank, 'No input detected', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Title',blank)
            # cv2.imshow('ROI', hand)
        else:
            # cv2.imshow('ROI', hand)
            prob,number=predict(hand,Myvggc2)
            if (torch.gt(prob,0.7)):
                cv2.putText(blank, 'predicted as '+str(number), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Title',blank)
                pub.publish(str(number))
            else:
                cv2.putText(blank, 'No input detected', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    index+=1
    t = time.time()
    tname = str(t)[5:10]
    c = cv2.waitKey(1)

    # 如果按下ESC则关闭窗口，同时跳出循环
    if c == 27:
        cv2.destroyAllWindows()
        break
