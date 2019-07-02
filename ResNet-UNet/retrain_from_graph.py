# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:14:29 2019

@author: ASabry
"""

from MIOU import MIOU, Kaggle_MIOU
import tensorflow as tf
from Data_Processing import training_data_augmentation_, get_data_folds_original
import numpy as np
from CheckPointMaker import make_checkpoint
import warnings

warnings.filterwarnings("ignore")

tf.reset_default_graph()
graph = tf.get_default_graph()

img_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Train\images'
mask_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Train\masks'
depths_path = r'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/depths.csv'
check_point_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Check Point/'
meta_graph_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Check Point/.meta'
every_ckpt_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Every Check Point/'
last_ckpt_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Last Check Point/'
x_train, y_train, x_test, y_test = training_data_augmentation_(img_path, mask_path, depths_path, img_size = 192, padding = 16, n_folds = 8, fold = 2)

x_train = x_train.reshape([-1, 224, 224, 1])
x_test = x_test.reshape([-1, 224, 224, 1])
y_train = y_train
y_test = y_test
batch_size = 32
epochs = 500
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
train_accuraciess =[]
test_accuraciess = []
losses = []
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
config.gpu_options.allow_growth = True

with tf.Session() as sess:    
    new_saver = tf.train.import_meta_graph(meta_graph_path)
    new_saver.restore(sess,tf.train.latest_checkpoint(check_point_path))
    early_stop = make_checkpoint(check_point_path, every_ckpt_path, last_ckpt_path, new_saver, sess, await_epochs = 200)
    
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    decay = graph.get_tensor_by_name('decay:0')
    LR = graph.get_tensor_by_name('LR:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    prediction = graph.get_tensor_by_name("Prediction:0")
    
    loss = graph.get_tensor_by_name("hinge_loss/value:0")
    opts = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_opt = graph.get_operation_by_name('optimization')
    opt = tf.group([train_opt, opts])

    for epoch in range(epochs):
        
        train_accuracies = []
        ktrain_accuracies = []
        test_accuracies = []
        ktest_accuracies = []
        
        print('epoch ', epoch+1, 'has started out of ', epochs, 'epochs')
        epoch_loss = 0
        start = 0
        end = batch_size
        
        for batch in range(len(x_train)//batch_size):
            
            print('Batch number', batch+1, 'has Started out of ', len(x_train)//batch_size, 'Batches', 'of epoch ', epoch+1)
            
            ex = x_train[start:end].astype(np.float16)
            ey = y_train[start:end].astype(np.float16)
            start += batch_size
            end += batch_size
            
            _, c= sess.run([opt, loss], feed_dict = {x: ex, y: ey, is_training: True, keep_prob: 0.5, decay: 0.98})

            epoch_loss += c
            
            p = prediction.eval(feed_dict = {x: ex, y: ey, is_training: False, keep_prob : 1.0, decay: 0.98}).astype(np.float32)
            ye = y.eval(feed_dict = {x: ex, y: ey, is_training: False, keep_prob : 1.0, decay: 0.98}).astype(np.float32)

            accuracy_train = MIOU(p, ye, 0.5, batch_size)
            kaccuracy_train = Kaggle_MIOU(p, ye, thresholds, batch_size)
            train_accuracies.append(accuracy_train)
            ktrain_accuracies.append(kaccuracy_train)
            
        start = 0
        end = batch_size
        for i in range(len(x_test)//batch_size):      
            
            xxx= x_test[start: end]
            yyy= y_test[start: end]
            start += batch_size
            end += batch_size
            
            p = prediction.eval(feed_dict = {x: xxx, y: yyy, is_training: False, keep_prob: 1.0, decay: 0.98}).astype(np.float32)
            ye = y.eval(feed_dict = {x: xxx, y: yyy, is_training: False, keep_prob: 1.0, decay: 0.98}).astype(np.float32)
            
            kaccuracy_test = Kaggle_MIOU(p, ye, thresholds, batch_size)
            accuracy_test = MIOU(p, ye, 0.5, batch_size)
            test_accuracies.append(accuracy_test)
            ktest_accuracies.append(kaccuracy_test)
            
        test_acc = np.array(test_accuracies)
        ktest_acc = np.array(ktest_accuracies)
        train_acc = np.array(train_accuracies)
        ktrain_acc = np.array(ktrain_accuracies)
        
        test_acc = np.mean(test_acc.ravel())
        ktest_acc = np.mean(ktest_acc.ravel())
        train_acc = np.mean(train_acc.ravel())
        ktrain_acc = np.mean(ktrain_acc.ravel())
        
        train_accuraciess.append((epoch+1, train_acc, ktrain_acc))
        test_accuraciess.append((epoch+1, test_acc, ktest_acc))
        losses.append(epoch_loss)
        early_stop.add(epoch+1, test_acc)
        print('epoch number ', epoch+1, 'has finished. train accuracy is: ', train_acc, 'kaggle acc is: ',ktrain_acc , 'test accuracy is: ', test_acc,'kaggle test is: ', ktest_acc, 'loss is: ', epoch_loss)
        if early_stop.end_epochs == True:
            break
        
