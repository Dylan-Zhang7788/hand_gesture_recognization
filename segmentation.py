#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import time
import cv2

from net.pspnet import PSPNet

models = {
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet')
}

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info(
            "Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        transforms.Resize((224, 224), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(transform_image_list)


def segment(net, img):
    # ------------ load image ------------ #
    data_transform = get_transform()
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = data_transform(img)
    img = img.cuda()

    # --------------- inference --------------- #
    with torch.no_grad():
        pred, _ = net(img.unsqueeze(dim=0))
        pred = pred.squeeze(dim=0)
        pred=torch.argmax(pred,axis=0)-14.5     # 14和15是手
        pred=torch.threshold(1.5-pred.abs(),0.8,0)*255
        final_graph=pred.expand(3,-1,-1)
        final_graph=final_graph.permute(1,2,0)
    return final_graph

def main():

    # --------------- Example of use --------------- #
    snapshot = os.path.join('./checkpoints', 'densenet', 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, 'densenet')
    net.eval()
    img = cv2.imread('1.jpg')
    cv2.imshow('1',img)
    cv2.waitKey(0)
    original_graph = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    original_graph = cv2.resize(original_graph, dsize=(224, 224))
    start = time.clock()
    mask = segment(net, img)
    end = time.clock()
    mask=mask.cpu().numpy().astype(np.uint8)
    cv2.imwrite('mask.jpg',mask)

    print('Prediction on batch took: {} s'.format(end - start))
    print()


if __name__ == '__main__':
    main()
