# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:09:57 2019

@author: ASabry
"""
import tensorflow as tf

#def scSE(input_layer, reduce_to_units = None):
#    ## Channel wise Squeeze and Excitation:
#    initializer = tf.contrib.layers.xavier_initializer_conv2d()
#
#    shapes = input_layer.get_shape().as_list()
#    avg_pooled = tf.nn.avg_pool(input_layer, 
#                                [1, shapes[1], shapes[2], 1], 
#                                [1, 1, 1, 1], 
#                                padding = 'VALID')
#    if reduce_to_units == None:
#        FC1 = tf.contrib.layers.fully_connected(avg_pooled, shapes[-1], activation_fn = tf.nn.relu)
#    else: FC1 = tf.contrib.layers.fully_connected(avg_pooled, reduce_to_units, activation_fn = tf.nn.sigmoid)
#    FC2 = tf.contrib.layers.fully_connected(FC1, shapes[-1], activation_fn = tf.nn.sigmoid)
#    final = input_layer * FC2
#    
#    ## Spacial wise Squeeze and Excitation:
#    weights = tf.Variable(initializer([1, 1, shapes[-1], 1]), dtype = tf.float32)
#    bias = tf.Variable(tf.zeros([1]))
#    conved_excitor = tf.nn.conv2d(input_layer, weights, [1, 1, 1, 1], 'VALID')
#    conved_excitor = tf.nn.sigmoid(conved_excitor+bias)
#    final_ = input_layer * conved_excitor
#    scse = final+final_
#    return scse


#
def scSE(input_layer, name, reduce_to_pct = 0.5):
    ## Channel wise Squeeze and Excitation:
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initializer_ = tf.contrib.layers.xavier_initializer()
    shapes = input_layer.get_shape().as_list()
    
    
    
    
    avg_pooled = tf.nn.avg_pool(input_layer, 
                                [1, shapes[1], shapes[2], 1], 
                                [1, 1, 1, 1], 
                                padding = 'VALID')
    avg_pooled_ = tf.reshape(avg_pooled, [shapes[0], shapes[3]])
    if reduce_to_pct == None:
        w1 = tf.Variable(initializer_([shapes[3], shapes[3]]), name = name+'w1')
        w2 = tf.Variable(initializer_([shapes[3], shapes[3]]), name = name+'w2')
        b1 = tf.Variable(tf.zeros([shapes[3]]), name = name+'b1')
        b2 = tf.Variable(tf.zeros([shapes[3]]), name = name+'b2')
        
        FC1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(avg_pooled_, w1), b1))
        FC2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(FC1, w2), b2))
    else: 
        w1_ = tf.Variable(initializer_([shapes[3], int(reduce_to_pct*shapes[3])]), name = name+'w1')
        w2_ = tf.Variable(initializer_([int(reduce_to_pct*shapes[3]), shapes[3]]), name = name+'w2')
        b1_ = tf.Variable(tf.zeros([int(reduce_to_pct*shapes[3])]), name = name+'b1')
        b2_ = tf.Variable(tf.zeros([shapes[3]]), name = name+'b2')
        
        FC1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(avg_pooled_, w1_), b1_))
        FC2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(FC1, w2_), b2_))
    FC3 = tf.reshape(FC2, [shapes[0], 1, 1, shapes[3]])
    final = input_layer * FC3
    
    ## Spacial wise Squeeze and Excitation:
    weights = tf.Variable(initializer([1, 1, shapes[-1], 1]), dtype = tf.float32, name = name+'11conv')
    bias = tf.Variable(tf.zeros([1]), name = name+'11bias')
    conved_excitor = tf.nn.conv2d(input_layer, weights, [1, 1, 1, 1], 'VALID')
    conved_excitor = tf.nn.sigmoid(tf.nn.bias_add(conved_excitor,bias))
    final_ = input_layer * conved_excitor
    scse = final+final_
    return scse


def scSE_Pretrained(input_layer, name, w1, w2, b1, b2, conv, bconv):
    shapes = input_layer.get_shape().as_list()
    avg_pooled = tf.nn.avg_pool(input_layer, 
                                [1, shapes[1], shapes[2], 1], 
                                [1, 1, 1, 1], 
                                padding = 'VALID')
    avg_pooled_ = tf.reshape(avg_pooled, [shapes[0], shapes[3]])
#    ww1 = tf.Variable(w1, name = name+'w1')
#    ww2 = tf.Variable(w2, name = name+'w2')
#    bb1 = tf.Variable(b1, name = name+'b1')
#    bb2 = tf.Variable(b2, name = name+'b2')
    
    FC1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(avg_pooled_, w1), b1))
    FC2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(FC1, w2), b2))
    
    FC3 = tf.reshape(FC2, [shapes[0], 1, 1, shapes[3]])
    final = input_layer * FC3
    
    ## Spacial wise Squeeze and Excitation:
#    weights = tf.Variable(conv, dtype = tf.float32, name = name+'11conv')
#    bias = tf.Variable(bconv, name = name+'11bias')
    conved_excitor = tf.nn.conv2d(input_layer, conv, [1, 1, 1, 1], 'VALID')
    conved_excitor = tf.nn.sigmoid(tf.nn.bias_add(conved_excitor,bconv))
    final_ = input_layer * conved_excitor
    scse = tf.math.maximum(final, final_)
    return scse




















