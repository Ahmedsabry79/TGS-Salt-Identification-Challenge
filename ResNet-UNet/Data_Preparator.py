# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:47:44 2019

@author: ASabry
"""

import numpy as np 
import pandas as pd
import random
import os
from Data_Preparation import training_data_generator
import cv2
import time
img_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Train\images'
mask_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Train\masks'
depths_path = 'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/depths.csv'

#def get_data(img_path, mask_path, depths_path, val_pct = 0.1):
#    data = training_data_generator(mask_path, img_path, 192, 16)
#    depths = pd.read_csv(depths_path)
#    
#    train = os.listdir(img_path)
#    for i in range(len(train)):
#        train[i] = train[i].replace('.png', '')
#        
#    depths_dict = {}
#    for i in depths.values:
#        if i[0] in train:
#            depths_dict[i[0]] = i[1]
#    
#    depth_1 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_2 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_3 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_4 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_5 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    categories = [depth_1, depth_2, depth_3, depth_4, depth_5]
#    
#    for i in data:
#        if np.sum(i[2]) == 0:
#            index = 'empty_masks'
#        if np.sum(i[2]) <= 250 and np.sum(i[2]) >0:
#            index = 'to_250'
#        if np.sum(i[2]) <= 2000 and np.sum(i[2]) >250:
#            index = 'to_2000'
#        if np.sum(i[2]) <= 4000 and np.sum(i[2]) >2000:
#            index = 'to_4000'
#        if np.sum(i[2]) <= 6000 and np.sum(i[2]) >4000:
#            index = 'to_6000'
#        if np.sum(i[2]) <= 8000 and np.sum(i[2]) >6000:
#            index = 'to_8000'
#        if np.sum(i[2]) <= 10000 and np.sum(i[2]) >8000:
#            index = 'to_10000'
#        if np.sum(i[2]) > 10000:
#            index = 'more_than_10000'
#            
#        if depths_dict[i[0]] in range(201): 
#            categories[0][index].append(i)
#        if depths_dict[i[0]] in range(201, 401): 
#            categories[1][index].append(i)
#        if depths_dict[i[0]] in range(401, 601): 
#            categories[2][index].append(i)
#        if depths_dict[i[0]] in range(601, 801): 
#            categories[3][index].append(i)
#        if depths_dict[i[0]] in range(801, 1001): 
#            categories[4][index].append(i)
#    
#    training = []
#    validation = []
#    
#    for cat in categories:
#        for index in cat:
#            forward = int(len(cat[index]) - (val_pct*len(cat[index])))
#            backward = int(len(cat[index])-forward)
#            for i in cat[index][:forward]:
#                training.append([i[1], i[2]])
#            for i in cat[index][-backward:]:
#                validation.append([i[1], i[2]])
#    
#    random.shuffle(training)
#    random.shuffle(training)
#    random.shuffle(training)
#    random.shuffle(validation)
#    random.shuffle(validation)
#    random.shuffle(validation)
#    
#    X_train = np.array([i[0] for i in training])
#    y_train = np.array([i[1] for i in training])
#    X_val = np.array([i[0] for i in validation])
#    y_val = np.array([i[1] for i in validation])
#    
#    return X_train, y_train, X_val, y_val
#    
    
#get_data(img_path, mask_path, depths_path, val_pct = 1/8)
    
    
def get_data_folds_original(img_path, mask_path, depths_path, img_size = 192, padding = 16, n_folds = 8, fold = 1):
    data = training_data_generator(mask_path, img_path, img_size, padding)
    depths = pd.read_csv(depths_path)
    
    train = os.listdir(img_path)
    for i in range(len(train)):
        train[i] = train[i].replace('.png', '')
        
    depths_dict = {}
    for i in depths.values:
        if i[0] in train:
            depths_dict[i[0]] = i[1]
    
    depth_1 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_2 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_3 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_4 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_5 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    categories = [depth_1, depth_2, depth_3, depth_4, depth_5]
    
    for i in data:
        if np.sum(i[2]) == 0:
            index = 'empty_masks'
        if np.sum(i[2]) <= 250 and np.sum(i[2]) >0:
            index = 'to_250'
        if np.sum(i[2]) <= 2000 and np.sum(i[2]) >250:
            index = 'to_2000'
        if np.sum(i[2]) <= 4000 and np.sum(i[2]) >2000:
            index = 'to_4000'
        if np.sum(i[2]) <= 6000 and np.sum(i[2]) >4000:
            index = 'to_6000'
        if np.sum(i[2]) <= 8000 and np.sum(i[2]) >6000:
            index = 'to_8000'
        if np.sum(i[2]) <= 10000 and np.sum(i[2]) >8000:
            index = 'to_10000'
        if np.sum(i[2]) > 10000:
            index = 'more_than_10000'
            
        if depths_dict[i[0]] in range(201): 
            categories[0][index].append(i)
        if depths_dict[i[0]] in range(201, 401): 
            categories[1][index].append(i)
        if depths_dict[i[0]] in range(401, 601): 
            categories[2][index].append(i)
        if depths_dict[i[0]] in range(601, 801): 
            categories[3][index].append(i)
        if depths_dict[i[0]] in range(801, 1001): 
            categories[4][index].append(i)
    
    training = []
    validation = []
    val_pct = 1/n_folds
    
    for cat in categories:
        for index in cat:
            idx = int(len(cat[index])*val_pct)
#            print('index is', idx)
            for i in cat[index][idx*(fold-1):idx*fold]:
#                print('total length is:', len(cat[index]))
#                print('val starts from', idx*(fold-1) ,'to', idx*fold)
                validation.append([i[1], i[2]])
#                print('train length is:', len(cat[index][idx*fold:])+len(cat[index][:idx*(fold-1)]))
            for i in cat[index][idx*fold:]:
                training.append([i[1], i[2]])
            for i in cat[index][:idx*(fold-1)]:
                training.append([i[1], i[2]])
    
    random.shuffle(training)
    random.shuffle(training)
    random.shuffle(training)
    random.shuffle(validation)
    random.shuffle(validation)
    random.shuffle(validation)
    
    X_train = np.array([i[0] for i in training])
    y_train = np.array([i[1] for i in training])
    X_val = np.array([i[0] for i in validation])
    y_val = np.array([i[1] for i in validation])
    
    return X_train, y_train, X_val, y_val


def get_data_folds(img_path, mask_path, depths_path, n_folds = 5, fold = 1, img_size = 192, pad = 16):
    data = training_data_generator(mask_path, img_path, img_size, pad)
    depths = pd.read_csv(depths_path)
    
    train = os.listdir(img_path)
    for i in range(len(train)):
        train[i] = train[i].replace('.png', '')
        
    depths_dict = {}
    for i in depths.values:
        if i[0] in train:
            depths_dict[i[0]] = i[1]
    
    depth_1 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_2 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_3 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_4 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    depth_5 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
    categories = [depth_1, depth_2, depth_3, depth_4, depth_5]
    
    for i in data:
        if np.sum(i[2]) == 0:
            index = 'empty_masks'
        if np.sum(i[2]) <= 250 and np.sum(i[2]) >0:
            index = 'to_250'
        if np.sum(i[2]) <= 2000 and np.sum(i[2]) >250:
            index = 'to_2000'
        if np.sum(i[2]) <= 4000 and np.sum(i[2]) >2000:
            index = 'to_4000'
        if np.sum(i[2]) <= 6000 and np.sum(i[2]) >4000:
            index = 'to_6000'
        if np.sum(i[2]) <= 8000 and np.sum(i[2]) >6000:
            index = 'to_8000'
        if np.sum(i[2]) <= 10000 and np.sum(i[2]) >8000:
            index = 'to_10000'
        if np.sum(i[2]) > 10000:
            index = 'more_than_10000'
            
        if depths_dict[i[0]] in range(201): 
            categories[0][index].append(i)
        if depths_dict[i[0]] in range(201, 401): 
            categories[1][index].append(i)
        if depths_dict[i[0]] in range(401, 601): 
            categories[2][index].append(i)
        if depths_dict[i[0]] in range(601, 801): 
            categories[3][index].append(i)
        if depths_dict[i[0]] in range(801, 1001): 
            categories[4][index].append(i)
    
    training = []
    validation = []
    val_pct = 1/n_folds
    
    for cat in categories:
        for index in cat:
            idx = int(len(cat[index])*val_pct)
#            print('index is', idx)
            for i in cat[index][idx*(fold-1):idx*fold]:
#                print('total length is:', len(cat[index]))
#                print('val starts from', idx*(fold-1) ,'to', idx*fold)
                validation.append([i[1], i[2]])
#                print('train length is:', len(cat[index][idx*fold:])+len(cat[index][:idx*(fold-1)]))
            for i in cat[index][idx*fold:]:
                if not i == None:
                    training.append(i[0])
            for i in cat[index][:idx*(fold-1)]:
                if not i == None:
                    training.append(i[0])

    random.shuffle(validation)
    random.shuffle(validation)
    random.shuffle(validation)
    
    
    return training, validation


def training_data_augmentation_(train_path, mask_path, depths_path, img_size= 192, padding = 16, n_folds = 5, fold = 1):
    training, validation = get_data_folds(train_path, mask_path, depths_path, n_folds = n_folds, fold = fold, img_size = img_size, pad = padding)
    Dataset = []
    imgs = [i+'.png' for i in training]
    print('Data preparation has started')
    a = 0
    start = 100
    for img in os.listdir(train_path):
        if a in range(100, len(imgs), 100):
            print(start, 'images are processed out of ', len(imgs), 'images')
            start+=100
        if img in imgs:
            original_img_101 = cv2.imread(os.path.join(train_path, img), 0)
            original_mask = cv2.imread(os.path.join(mask_path, img), 0)
            original_mask[original_mask>0]= 1
            original_img_192 = cv2.resize(original_img_101.copy(), (img_size, img_size))
            original_img_224 = cv2.copyMakeBorder(original_img_192.copy(), padding, padding, padding, padding, 0, value= 0)
            Dataset.append([original_img_224, original_mask, 0])
            img = img.replace('.png', '')
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/3.png', original_img_224)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/4.png', np.where(original_mask==1, 255, 0))
            
            
            ## Flipping Horizontally:
            flipped_original_224 = cv2.flip(original_img_224.copy(), 1)
            flipped_mask = cv2.flip(original_mask, 1)
            Dataset.append([flipped_original_224, flipped_mask, 1])
#            print(type(flipped_original_224))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/5.png', flipped_original_224)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/6.png', np.where(flipped_mask==1, 255, 0))
#            time.sleep(5)
            
            
            ## Flipping 2:
            flipped_original_224_2 = cv2.flip(original_img_224.copy(), 0)
            flipped_mask_2 = cv2.flip(original_mask.copy(), 0)
            Dataset.append([flipped_original_224_2, flipped_mask_2, 2])
#            print(type(flipped_original_224_2))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/7.png', flipped_original_224_2)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/8.png', np.where(flipped_mask_2==1, 255, 0))
#            time.sleep(5)
            
            
            ## Flipping 3:
#            flipped_original_224_3 = cv2.flip(original_img_224.copy(), -1)
#            flipped_mask_3 = cv2.flip(original_mask.copy(), -1)
#            Dataset.append([flipped_original_224_3, flipped_mask_3, 3])
#            print(type(flipped_original_224_3))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/9.png', flipped_original_224_3)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/10.png', np.where(flipped_mask_3==1, 255, 0))
#            time.sleep(5)
            
            
            ## 90 degrees rotation:
#            rotated_original_224 = cv2.rotate(original_img_224.copy(), 2)
#            rotated_mask = cv2.rotate(original_mask.copy(), 2)
#            Dataset.append([rotated_original_224, rotated_mask, 4])
#            print(type(rotated_original_224))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/15.png', rotated_original_224)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/16.png', np.where(rotated_mask==1, 255, 0))
#            time.sleep(5)
            
            
            ## Random Crops 1:
#            Rx = random.randint(0, 64)
#            Ry = random.randint(0, 64)
#    
#            original_img_224_crop1 = cv2.resize(original_img_101.copy(), (256, 256))
#            mask_img_224_crop1 = cv2.resize(original_mask.copy(), (256, 256))
#            
#            C_original_img = original_img_224_crop1[Ry:Ry+192, Rx: Rx+192]
#            C_mask_img = mask_img_224_crop1[Ry:Ry+192, Rx: Rx+192]
#    
#            C_original_img_224 =cv2.resize(C_original_img, (img_size, img_size))
#            C_original_img_224_ = cv2.copyMakeBorder(C_original_img_224.copy(), padding, padding, padding, padding, 0, value= 0)
#            C_mask_img_101 = cv2.resize(C_mask_img, (101, 101))
#    
#            Dataset.append([C_original_img_224_, C_mask_img_101, 5])
#            print(type(C_original_img_224_))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/11.png', C_original_img_224_)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/12.png', np.where(C_mask_img_101==1, 255, 0))
#            time.sleep(5)
            
            
            ## Random Crops 2:
            Rx = random.randint(0, 128)
            Ry = random.randint(0, 128)
    
            original_img_224_crop2 = cv2.resize(original_img_101.copy(), (256, 256))
            mask_img_224_crop2 = cv2.resize(original_mask.copy(), (256, 256))
            
            C_original_img_1 = original_img_224_crop2[Ry:Ry+128, Rx: Rx+128]
            C_mask_img_1 = mask_img_224_crop2[Ry:Ry+128, Rx: Rx+128]
    
            C_original_img_2241 =cv2.resize(C_original_img_1, (img_size, img_size))
            C_original_img_224_1 = cv2.copyMakeBorder(C_original_img_2241.copy(), padding, padding, padding, padding, 0, value= 0)
            C_mask_img_101 = cv2.resize(C_mask_img_1, (101, 101))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/13.png', C_original_img_224_1)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/14.png', np.where(C_mask_img_101==1, 255, 0))
#            time.sleep(5)
            Dataset.append([C_original_img_224_1, C_mask_img_101, 3])
            
            ## Invert intensity:
            inverted_192 = 255 - original_img_192.copy()
            inverted_224 = cv2.copyMakeBorder(inverted_192.copy(), padding, padding, padding, padding, 0, value= 0)
            Dataset.append([inverted_224, original_mask, 4])
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/17.png', inverted_224)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/18.png', np.where(original_mask==1, 255, 0))
#            time.sleep(5)
            
            # Random_crops:
#            Rx = random.randint(0, 96)
#            Ry = random.randint(0, 96)
#    
#            original_img_224_crop1_ii = cv2.resize(inverted_192.copy(), (256, 256))
#            mask_img_224_crop1_ii = cv2.resize(original_mask.copy(), (256, 256))
#            
#            C_original_img_ii = original_img_224_crop1_ii[Ry:Ry+160, Rx: Rx+160]
#            C_mask_img_ii = mask_img_224_crop1_ii[Ry:Ry+160, Rx: Rx+160]
#    
#            C_original_img_224_ii =cv2.resize(C_original_img_ii, (img_size, img_size))
#            C_original_img_224_ii = cv2.copyMakeBorder(C_original_img_224_ii.copy(), padding, padding, padding, padding, 0, value= 0)
#            C_mask_img_101_ii = cv2.resize(C_mask_img_ii, (101, 101))
#    
#            Dataset.append([C_original_img_224_ii, C_mask_img_101_ii, 5])
#            
            ## Flipping 3:
            flipped_original_224_3i = cv2.flip(inverted_224.copy(), -1)
            flipped_mask_3i = cv2.flip(original_mask.copy(), -1)
            Dataset.append([flipped_original_224_3i, flipped_mask_3i, 6])
#            print(type(flipped_original_224_3))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/9.png', flipped_original_224_3)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/10.png', np.where(flipped_mask_3==1, 255, 0))
#            time.sleep(5)
            
            
            ## 90 degrees rotation:
            rotated_original_224i = cv2.rotate(inverted_224.copy(), 2)
            rotated_maski = cv2.rotate(original_mask.copy(), 2)
            Dataset.append([rotated_original_224i, rotated_maski, 7])
#            print(type(rotated_original_224))
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/15.png', rotated_original_224)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/16.png', np.where(rotated_mask==1, 255, 0))
#            time.sleep(5)
            a+=1
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/13.png', C_original_img_224_1)
#            cv2.imwrite('C:/Users/MSabry/Desktop/New folder (2)/14.png', np.where(C_mask_img_101==1, 255, 0))
#            time.sleep(5)
#            print(type(C_original_img_224_1))
    random.shuffle(Dataset)
    random.shuffle(Dataset)
    random.shuffle(Dataset)
    X_train = np.array([i[0] for i in Dataset])
    y_train = np.array([i[1] for i in Dataset])
    X_val = np.array([i[0] for i in validation])
    y_val = np.array([i[1] for i in validation])
    
    print('Data preparation has ended')
    return X_train, y_train, X_val, y_val
    
    
#def get_augmented_data_folds(img_path, mask_path, depths_path, n_folds = 8, fold = 1):
#    data = training_data_augmentation(mask_path, img_path, 192, 16)
#    depths = pd.read_csv(depths_path)
#    
#    train = os.listdir(img_path)
#    for i in range(len(train)):
#        train[i] = train[i].replace('.png', '')
#        
#    depths_dict = {}
#    for i in depths.values:
#        if i[0] in train:
#            depths_dict[i[0]] = i[1]
#    
#    depth_1 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_2 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_3 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_4 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_5 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    categories = [depth_1, depth_2, depth_3, depth_4, depth_5]
#    
#    for i in data:
#        if np.sum(i[2]) == 0:
#            index = 'empty_masks'
#        if np.sum(i[2]) <= 250 and np.sum(i[2]) >0:
#            index = 'to_250'
#        if np.sum(i[2]) <= 2000 and np.sum(i[2]) >250:
#            index = 'to_2000'
#        if np.sum(i[2]) <= 4000 and np.sum(i[2]) >2000:
#            index = 'to_4000'
#        if np.sum(i[2]) <= 6000 and np.sum(i[2]) >4000:
#            index = 'to_6000'
#        if np.sum(i[2]) <= 8000 and np.sum(i[2]) >6000:
#            index = 'to_8000'
#        if np.sum(i[2]) <= 10000 and np.sum(i[2]) >8000:
#            index = 'to_10000'
#        if np.sum(i[2]) > 10000:
#            index = 'more_than_10000'
#            
#        if depths_dict[i[0]] in range(201): 
#            categories[0][index].append(i)
#        if depths_dict[i[0]] in range(201, 401): 
#            categories[1][index].append(i)
#        if depths_dict[i[0]] in range(401, 601): 
#            categories[2][index].append(i)
#        if depths_dict[i[0]] in range(601, 801): 
#            categories[3][index].append(i)
#        if depths_dict[i[0]] in range(801, 1001): 
#            categories[4][index].append(i)
#    
#    training = []
#    validation = []
#    val_pct = 1/n_folds
#    
#    for cat in categories:
#        for index in cat:
#            idx = int(len(cat[index])*val_pct)
##            print('index is', idx)
#            for i in cat[index][idx*(fold-1):idx*fold]:
##                print('total length is:', len(cat[index]))
##                print('val starts from', idx*(fold-1) ,'to', idx*fold)
#                validation.append([i[1], i[2]])
##                print('train length is:', len(cat[index][idx*fold:])+len(cat[index][:idx*(fold-1)]))
#            for i in cat[index][idx*fold:]:
#                training.append([i[1], i[2]])
#            for i in cat[index][:idx*(fold-1)]:
#                training.append([i[1], i[2]])
#    
#    random.shuffle(training)
#    random.shuffle(training)
#    random.shuffle(training)
#    random.shuffle(validation)
#    random.shuffle(validation)
#    random.shuffle(validation)
#    
#    X_train = np.array([i[0] for i in training])
#    y_train = np.array([i[1] for i in training])
#    X_val = np.array([i[0] for i in validation])
#    y_val = np.array([i[1] for i in validation])
#    
#    return X_train, y_train, X_val, y_val
#    
    
#def get_augmented_data_folds(img_path, mask_path, depths_path, n_val = 500, n_train = 4000):
#    data = training_data_augmentation(mask_path, img_path, 192, 16)
#    depths = pd.read_csv(depths_path)
#    
#    train = os.listdir(img_path)
#    for i in range(len(train)):
#        train[i] = train[i].replace('.png', '')
#        
#    depths_dict = {}
#    for i in depths.values:
#        if i[0] in train:
#            depths_dict[i[0]] = i[1]
#    
#    depth_1 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_2 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_3 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_4 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    depth_5 = {'empty_masks': [], 'to_250': [], 'to_2000': [], 'to_4000': [], 'to_6000': [], 'to_8000': [], 'to_10000': [], 'more_than_10000': []}
#    categories = [depth_1, depth_2, depth_3, depth_4, depth_5]
#    
#    for i in data:
#        if np.sum(i[2]) == 0:
#            index = 'empty_masks'
#        if np.sum(i[2]) <= 250 and np.sum(i[2]) >0:
#            index = 'to_250'
#        if np.sum(i[2]) <= 2000 and np.sum(i[2]) >250:
#            index = 'to_2000'
#        if np.sum(i[2]) <= 4000 and np.sum(i[2]) >2000:
#            index = 'to_4000'
#        if np.sum(i[2]) <= 6000 and np.sum(i[2]) >4000:
#            index = 'to_6000'
#        if np.sum(i[2]) <= 8000 and np.sum(i[2]) >6000:
#            index = 'to_8000'
#        if np.sum(i[2]) <= 10000 and np.sum(i[2]) >8000:
#            index = 'to_10000'
#        if np.sum(i[2]) > 10000:
#            index = 'more_than_10000'
#            
#        if depths_dict[i[0]] in range(201): 
#            categories[0][index].append(i)
#        if depths_dict[i[0]] in range(201, 401): 
#            categories[1][index].append(i)
#        if depths_dict[i[0]] in range(401, 601): 
#            categories[2][index].append(i)
#        if depths_dict[i[0]] in range(601, 801): 
#            categories[3][index].append(i)
#        if depths_dict[i[0]] in range(801, 1001): 
#            categories[4][index].append(i)
#    
#    training = []
#    validation = []
#    
#    for cat in categories:
#        for index in cat:
#            original = []
#            for i in cat[index]:
#                if i[3] != 0:
#                    training.append([i[1], i[2]])
#                else:
#                    original.append([i[1], i[2]])
#            for i in original[:n_val]:
#                print('original_length is:', len(original))
#                validation.append([i[1], i[2]])
#            for i in original[n_val:]:
#                training.append([i[1], i[2]])
#    
#    random.shuffle(training)
#    random.shuffle(training)
#    random.shuffle(training)
#    random.shuffle(validation)
#    random.shuffle(validation)
#    random.shuffle(validation)
#    
#    X_train = np.array([i[0] for i in training])
#    y_train = np.array([i[1] for i in training])
#    X_val = np.array([i[0] for i in validation])
#    y_val = np.array([i[1] for i in validation])
#    
#    return X_train, y_train, X_val, y_val
#    
#    