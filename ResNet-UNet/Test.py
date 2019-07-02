# -*- coding: utf-8 -*-
"""
Created on Sat May 18 01:37:55 2019

@author: ASabry
"""
import pandas as pd
import tensorflow as tf
from Data_Preparation import test_data_generator, encode_rle
import numpy as np

tf.reset_default_graph()
graph = tf.get_default_graph()

check_point_path = r'C:/Users/MSabry/Desktop/TGS Project/UNet224/Check Point/'
meta_graph_path = r'C:/Users/MSabry/Desktop/TGS Project/UNet224/Check Point/.meta'
test_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Test\images'
submission_path = r'C:/Users/MSabry/Desktop/sample_submission_1.csv'


test_data = test_data_generator(test_path, 192, 16)
M = [[i[0], i[1]/255.0] for i in test_data]
df = pd.DataFrame(columns = ['id', 'rle_mask'], index = pd.RangeIndex(start=0, stop=len(test_data)+16, step=1))

batch_size = 32

with tf.Session() as sess:    
    new_saver = tf.train.import_meta_graph(meta_graph_path)
    new_saver.restore(sess,tf.train.latest_checkpoint(check_point_path))
    x = graph.get_tensor_by_name('x:0')
#    decay = graph.get_tensor_by_name('decay:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    prediction = graph.get_tensor_by_name("Prediction:0")
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    start = 0
    end = batch_size
    print('Started writing in the file')
    for batch in range(int(len(M)//batch_size)+1):
        if batch == int(len(M)//batch_size):
            bx = M[start: end]+M[:16]
        else:
            bx = M[start: end]
        indices = [i[0] for i in bx]
        imgs = [i[1].reshape(224, 224, 1) for i in bx]
        pred = prediction.eval(feed_dict = {x: imgs, is_training: True, keep_prob: 1.0})
        a = pred.copy()
        b = np.where(a>=0.5, 1, 0)
        submit = encode_rle(b, True, batch_size)
        df['id'][start: end] = indices
        df['rle_mask'][start: end] = submit
        start+= batch_size
        end += batch_size
        print('batch num', batch+1, 'has ended out of', int(18000/batch_size)+1, 'batches')
    df = df[:-16]
df.to_csv(submission_path, index = False)
print('Done')

















