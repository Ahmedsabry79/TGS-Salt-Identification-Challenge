import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from s_c_S_E import scSE

def Batch_Norm(x, is_training, decay = 0.9, scale = True, zero_debias = False):
    return tf.contrib.layers.batch_norm(x,
                                        decay = decay, 
                                        scale = scale, 
                                        is_training = is_training, 
                                        zero_debias_moving_mean = zero_debias)

def conv(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')

def convv(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='VALID')

def conv2s(x, w):
    return tf.nn.conv2d(x, w, [1, 2, 2, 1], padding='VALID')

def residual_block(x, nfilters, is_training, decay = 0.9, scale = True, zero_debias = False):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    w1 = tf.Variable(initializer([3, 3, nfilters, nfilters]))
    w2 = tf.Variable(initializer([3, 3, nfilters, nfilters]))
    b1 = tf.Variable(tf.zeros([nfilters]))
    b2 = tf.Variable(tf.zeros([nfilters]))

    L1 = Batch_Norm(x, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L2 = tf.nn.relu(L1)
    L3 = tf.nn.bias_add(conv(L2, w1), b1)

    L4 = Batch_Norm(L3, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L5 = tf.nn.relu(L4)
    L6 = tf.nn.bias_add(conv(L5, w2), b2)

    L7 = L6 * 0.5 + x * 0.5

    return L7  
    
def first_block(x, nfilters):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    w1 = tf.Variable(initializer([7, 7, 1, nfilters]))
    b1 = tf.Variable(tf.zeros([nfilters]))
    w2 = tf.Variable(initializer([2, 2, nfilters, nfilters]))
    b2 = tf.Variable(tf.zeros([nfilters]))

    L1 = tf.nn.relu(tf.nn.bias_add(conv(x, w1), b1))
    L2 = tf.nn.relu(tf.nn.bias_add(conv2s(L1, w2), b2))
    return L2

def intermediate_layer(x):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    shape = x.get_shape().as_list()
    c_in = shape[-1]
    c_out = shape[-1]*2

    w1 = tf.Variable(initializer([2, 2, c_in, c_out]))
    b1 = tf.Variable(tf.zeros([c_out]))

    L1 = tf.nn.relu(tf.nn.bias_add(conv2s(x, w1), b1))

    return L1

def Upsample_Block(x, is_training , decay , scale , zero_debias):
    c = x.get_shape().as_list()[-1]
    batch_size = x.get_shape().as_list()[0]
    size = x.get_shape().as_list()[1]
    
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    w1 = tf.Variable(initializer([2, 2, c//2, c]))
    w2 = tf.Variable(initializer([3, 3, c//2, c//2]))
    w3 = tf.Variable(initializer([3, 3, c//2, c//2]))
    
    b2 = tf.Variable(tf.zeros([c//2]))
    b3 = tf.Variable(tf.zeros([c//2]))
    
    L1 = tf.nn.conv2d_transpose(x, w1, output_shape = [batch_size, size*2, size*2, c//2], strides = [1, 2, 2, 1], padding = "VALID")
    L2 = Batch_Norm(L1, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L3 = tf.nn.relu(L2)
    
    L4 = tf.nn.bias_add(conv(L3, w2), b2)
    L5 = Batch_Norm(L4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L6 = tf.nn.relu(L5)
    
    L7 = tf.nn.bias_add(conv(L6, w3), b3)
    L8 = Batch_Norm(L7, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L9 = tf.nn.relu(L8)
    return L9

def Final_Upsample_Block(x, is_training , decay , scale , zero_debias):
    c = x.get_shape().as_list()[-1]
    batch_size = x.get_shape().as_list()[0]
    size = x.get_shape().as_list()[1]
    
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    w1 = tf.Variable(initializer([2, 2, c//2, c]))
    w2 = tf.Variable(initializer([3, 3, c//2, c//2]))
    w3 = tf.Variable(initializer([3, 3, c//2, 1]))
    
    b2 = tf.Variable(tf.zeros([c//2]))
    
    L1 = tf.nn.conv2d_transpose(x, w1, [batch_size, size*2, size*2, c//2], [1, 2, 2, 1], "VALID")
    L2 = Batch_Norm(L1, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L3 = tf.nn.relu(L2)
    
    L4 = tf.nn.bias_add(conv(L3, w2), b2)
    L5 = Batch_Norm(L4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L6 = tf.nn.relu(L5)
    
    L7 = conv(L6, w3)
    
    return L7

def ResNet34(x, initial_filters, is_training, decay, scale, zero_debias):
    
    ## Initial Layer:
    L1 = first_block(x, initial_filters)

    ## First Block Group:
    L2 = residual_block(L1, initial_filters, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L3 = residual_block(L2, initial_filters, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L4 = residual_block(L3, initial_filters, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L4 = scSE(L4)._scSE_()
    ## First Intermediate Layer:
    
    L5 = intermediate_layer(L4)

    ## Second Group Block:

    L6 = residual_block(L5, initial_filters*2, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L7 = residual_block(L6, initial_filters*2, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L8 = residual_block(L7, initial_filters*2, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L9 = residual_block(L8, initial_filters*2, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L9 = scSE(L9)._scSE_()
    
    ## Second Intermediate Layer:

    L10 = intermediate_layer(L9)

    ## Third Group Block:
    
    L11 = residual_block(L10, initial_filters*4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L12 = residual_block(L11, initial_filters*4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L13 = residual_block(L12, initial_filters*4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L14 = residual_block(L13, initial_filters*4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L15 = residual_block(L14, initial_filters*4, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L15 = scSE(L15)._scSE_()
    ## Third Intermediate Layer:

    L16 = intermediate_layer(L15)

    ## Fourth Group Block:
    
    L17 = residual_block(L16, initial_filters*8, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L18 = residual_block(L17, initial_filters*8, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    L19 = residual_block(L18, initial_filters*8, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)

    return L19

def last_decoder_layer(x):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    c = x.get_shape().as_list()[-1]
    
    w1 = tf.Variable(initializer([7, 7, c, c]))
    b1 = tf.Variable(tf.zeros([c]))
    
    w2 = tf.Variable(initializer([4, 4, c, c]))
    b2 = tf.Variable(tf.zeros([c]))
    
    w3 = tf.Variable(initializer([3, 3, c, 1]))
    
    L1 = tf.nn.relu(tf.nn.bias_add(convv(x, w1), b1))
    L2 = tf.nn.relu(tf.nn.bias_add(convv(L1, w2), b2))
    L3 = convv(L2, w3)
    return L3
    

def Decoder(x, is_training , decay , scale , zero_debias):
    
    L1 = Upsample_Block(x, is_training = is_training , decay = decay , scale = scale , zero_debias = zero_debias)
    L1 = scSE(L1)._scSE_()
    
    
    L2 = Upsample_Block(L1, is_training = is_training , decay = decay , scale = scale , zero_debias = zero_debias)
    L2 = scSE(L2)._scSE_()
    
    L3 = Upsample_Block(L2, is_training = is_training , decay = decay , scale = scale , zero_debias = zero_debias)
    L3 = scSE(L3)._scSE_()
    
    L4 = last_decoder_layer(L3)#, is_training = is_training , decay = decay , scale = scale , zero_debias = zero_debias)
#    L4 = scSE(L4)._scSE_()
#    
#    L5 = Upsample_Block(L4, is_training = is_training , decay = decay , scale = scale , zero_debias = zero_debias)
#    L5 = scSE(L5)._scSE_()
#    
#    L6 = Final_Upsample_Block(L5, is_training = is_training , decay = decay , scale = scale , zero_debias = zero_debias)

    
    return L4

    
#x = tf.placeholder(shape = [32, 1024, 1024, 1], dtype = tf.float32)
##
#y = ResNet34(x, 8, True, 0.7, True, False)
#z = Decoder(y, True, 0.7, True, False)











    

    






















    
