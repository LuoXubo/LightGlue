#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   estimate.py
@Time    :   2023/07/03 10:37:00
@Author  :   Xubo Luo 
@Version :   1.0
@Contact :   luoxubo@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   Batch localization
'''


from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, match_pair, transformation, coords, sigmoid, draw_matches
from lightglue import viz2d
from pathlib import Path
import torch
import time
import numpy as np
import os
from tqdm import tqdm
import cv2
# images = Path('uavimg')


extractor = SuperPoint(max_num_keypoints=2048, nms_radius=3).eval().cuda()  # load the extractor
match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval().cuda()

central_coords_x = 500
central_coords_y = 500
pt_drone = np.matrix([int(central_coords_x/2), int(central_coords_y/2), 1])

root_dataset = '/home/zino/lxb/Image-Matching-codes/datasets/dataset8/'
res_XY = root_dataset + 'uav/res_lightglue.txt'
gt_XY = root_dataset + 'uav/gt.txt'
save_path = root_dataset + 'lightglue_matches/'
root_images = root_dataset + 'lpn_results/'

uav_folder_list = os.listdir(root_images)
uav_folder_list.sort()

with open(gt_XY, 'r') as f:
    gtlines = f.readlines()
f.close()

cnt = 0
ts = 0.0

for uav_folder in tqdm(uav_folder_list):
    imgs = os.listdir(root_images + uav_folder)
    imgs.sort()
    best_score = 0
    best_mtch = 0
    uav = root_images + '/' + uav_folder + '/' + imgs[10]
    img1 = cv2.imread(uav)
    img_A = np.array(img1)
    tensorA, scalesA = load_image(uav, grayscale=False)

    subimg_candidate = imgs[0:10]

    start = time.time()
    print('Begin to search best sub-satellite imge for uav image: %s ......'%uav_folder)

    for satellite_folder in subimg_candidate: # satellite_folder: Rank01-45_1000_1000.png
        sat = root_images + '/'  + uav_folder + '/' + satellite_folder
        tensorB, scalesB = load_image(sat, grayscale=False)
        
        pred = match_pair(extractor, matcher, tensorA, tensorB)
        kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        if(len(m_kpts0) < 10):
            continue
        H, _ = cv2.findHomography(m_kpts0.numpy(), m_kpts1.numpy(), cv2.RANSAC, 5.0)

        matching_score = sigmoid(len(matches)/100)
        if matching_score > best_score:
            best_kpA = m_kpts0.numpy()
            best_kpB = m_kpts1.numpy()
            best_mtch = len(matches)
            best_score = matching_score
            best_sub = satellite_folder
            best_H = H

    # print(best_H)
    startX, startY = best_sub.split('.')[0].split('_')[1:3]
    pt_sate = transformation(pt_drone.T, best_H)

    x, y = coords(pt_sate)


    resX, resY = float(startX) + coords(pt_sate)[0], float(startY) + coords(pt_sate)[1]

    gtline = gtlines[cnt].split('\n')[0]
    timestamp = gtline.split(' ')[0] # 直接使用gt.txt里已经除以500后的时间戳
    z, qw, qx, qy, qz = gtline.split(' ')[3:]
    cnt = cnt + 1

    im2 = cv2.imread(root_images + '/'  + uav_folder + '/' + best_sub)
    img_B = np.array(im2)
    
    warped_uav_img = cv2.warpPerspective(img_A, best_H, (img_B.shape[0], img_B.shape[1]))
    added_img = cv2.addWeighted(warped_uav_img, 0.7, img_B, 0.3, 0)
    
    cv2.imwrite(save_path + uav_folder + '.png', added_img)

    cv2.imwrite(save_path + uav_folder + '_matches.png', draw_matches(img_A, img_B, best_kpA, best_kpB))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    end = time.time()
    print('Timestamp: %f        Position:{%f, %f}       matching points: %d     matchability:%f     timecost:%f s'%(ts, resX, resY, best_mtch, best_score, (end-start)))
    ts += 0.1
    

