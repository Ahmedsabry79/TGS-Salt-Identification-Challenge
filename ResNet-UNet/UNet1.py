# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:19:19 2019

@author: ASabry
"""

import tensorflow as tf
from scSE import scSE
#from Batch_Normalizations import Batch_Normalization

def Forward_UNet(x, is_training, batch_size, initial_filters = 64, apply_dropout = False, keep_prob = 0.5, batch_normalize = False):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    weights = {}
    biases = {}
    weights['L1a'] = tf.Variable(initializer([3, 3, 1, initial_filters]), name = 'wL1a')
    weights['L1b'] = tf.Variable(initializer([3, 3, initial_filters, initial_filters]), name = 'wL1b')
    weights['L2a'] = tf.Variable(initializer([3, 3, initial_filters, initial_filters*2]), name = 'wL2a')
    weights['L2b'] = tf.Variable(initializer([3, 3, initial_filters*2, initial_filters*2]), name = 'wL2b')
    weights['L3a'] = tf.Variable(initializer([3, 3, initial_filters*2, initial_filters*4]), name = 'wL3a')
    weights['L3b'] = tf.Variable(initializer([3, 3, initial_filters*4, initial_filters*4]), name = 'wL3b')
    weights['L4a'] = tf.Variable(initializer([3, 3, initial_filters*4, initial_filters*8]), name = 'wL4a')
    weights['L4b'] = tf.Variable(initializer([4, 4, initial_filters*8, initial_filters*8]), name = 'wL4b')
    weights['L5a'] = tf.Variable(initializer([2, 2, initial_filters*8, initial_filters*16]), name = 'wL5a')
    weights['L5b'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*16]), name = 'wL5b')  
    weights['L6a'] = tf.Variable(initializer([2, 2, initial_filters*16, initial_filters*16]), name = 'wL6a')
    weights['L6b'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*16]), name = 'wL6b')
    weights['L6c'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*16]), name = 'wL6c')
    weights['L7a'] = tf.Variable(initializer([2, 2, initial_filters*8, initial_filters*16]), name = 'wL7a')
    weights['L7b'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*8]), name = 'wL7b')
    weights['L7c'] = tf.Variable(initializer([3, 3, initial_filters*8, initial_filters*8]), name = 'wL7c')
    weights['L8a'] = tf.Variable(initializer([2, 2, initial_filters*4, initial_filters*8]), name = 'wL8a')
    weights['L8b'] = tf.Variable(initializer([3, 3, initial_filters*8, initial_filters*4]), name = 'wL8b')
    weights['L8c'] = tf.Variable(initializer([3, 3, initial_filters*4, initial_filters*4]), name = 'wL8c')
    weights['L9a'] = tf.Variable(initializer([2, 2, initial_filters*2, initial_filters*4]), name = 'wL9a')
    weights['L9b'] = tf.Variable(initializer([3, 3, initial_filters*4, initial_filters*2]), name = 'wL9b')
    weights['L9c'] = tf.Variable(initializer([3, 3, initial_filters*2, initial_filters*2]), name = 'wL9c')
    weights['L10a'] = tf.Variable(initializer([2, 2, initial_filters, initial_filters*2]), name = 'wL10a')
    weights['L10b'] = tf.Variable(initializer([2, 2, initial_filters*2, initial_filters]), name = 'wL10b')
    weights['L10c'] = tf.Variable(initializer([3, 3, initial_filters, initial_filters]), name = 'wL10c')
    weights['L10d'] = tf.Variable(initializer([1, 1, initial_filters, 1]), name = 'wL10d')
    
    biases['L1a'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL1a')
    biases['L1b'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL1b')
    biases['L2a'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL2a')
    biases['L2b'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL2b')
    biases['L3a'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL3a')
    biases['L3b'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL3b')
    biases['L4a'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL4a')
    biases['L4b'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL4b')
    biases['L5a'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL5a')
    biases['L5b'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL5b')  
    biases['L6a'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL6a')
    biases['L6b'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL6b')
    biases['L6c'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL6c')
    biases['L7a'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL7a')
    biases['L7b'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL7b')
    biases['L7c'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL7c')
    biases['L8a'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL8a')
    biases['L8b'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL8b')
    biases['L8c'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL8c')
    biases['L9a'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL9a')
    biases['L9b'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL9b')
    biases['L9c'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL9c')
    biases['L10a'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL10a')
    biases['L10b'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL10b')
    biases['L10c'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL10c')
    biases['L10d'] = tf.Variable(tf.zeros([1]), name = 'bL10d')
    
    ## Block of layers number 1:
    L1a = tf.nn.bias_add(conv_valid(x, weights['L1a']), biases['L1a'])
    if batch_normalize:
        L1b = Batch_Norm(L1a, is_training = is_training)
        L1b = tf.nn.relu(L1b)
        L1b = tf.nn.bias_add(conv_valid(L1b, weights['L1b']), biases['L1b'])
        crop_1 = L1b[:, 58: 58+104, 58: 58+104, :]
        L1b = tf.nn.relu(Batch_Norm(L1b, is_training = is_training))
    else: 
        L1a = tf.nn.relu(L1a)
        L1b = tf.nn.bias_add(conv_valid(L1a, weights['L1b']), biases['L1b'])
        crop_1 = L1b[:, 58: 58+104, 58: 58+104, :]
        L1b = tf.nn.relu(L1b)
    
    

    L1b = scSE(L1b, name = 'L1')
    L1c = pool(L1b)
    L1c = tf.nn.dropout(L1c, keep_prob)
    
    ## Block of layers number 2:
    L2a = tf.nn.bias_add(conv_valid(L1c, weights['L2a']), biases['L2a'])
    if batch_normalize:
        L2a = Batch_Norm(L2a, is_training = is_training)
        L2a = tf.nn.relu(L2a)
        L2b = tf.nn.bias_add(conv_valid(L2a, weights['L2b']), biases['L2b'])
        crop_2 = L2b[:, 25: 25+56, 25: 25+56, :]
        L2b = tf.nn.relu(Batch_Norm(L2b, is_training = is_training))
        
    else:
        L2a = tf.nn.relu(L2a)
        L2b = tf.nn.relu(tf.nn.bias_add(conv_valid(L2a, weights['L2b']), biases['L2b']))
        crop_2 = L2b[:, 25: 25+56, 25: 25+56, :]

    L2b = scSE(L2b, name = 'L2')
    L2c = pool(L2b)
    L2c = tf.nn.dropout(L2c, keep_prob)
    
    ## Block of layers number 3:
    L3a = tf.nn.bias_add(conv_valid(L2c, weights['L3a']), biases['L3a'])
    if batch_normalize:
        L3a = tf.nn.relu(Batch_Norm(L3a, is_training = is_training))
        L3b = tf.nn.bias_add(conv_valid(L3a, weights['L3b']), biases['L3b'])
        crop_3 = L3b[:, 8: 8+32, 8: 8+32, :]
        L3b = tf.nn.relu(Batch_Norm(L3a, is_training = is_training))
    else:
        L3a = tf.nn.relu(L3a)
        L3b = tf.nn.relu(tf.nn.bias_add(conv_valid(L3a, weights['L3b']), biases['L3b']))
        crop_3 = L3b[:, 8: 8+32, 8: 8+32, :]
    L3b = scSE(L3b, name = 'L3')
    L3c = pool(L3b)
    L3c = tf.nn.dropout(L3c, keep_prob)
    
    ## Block of layers number 4:
    L4a = tf.nn.bias_add(conv_valid(L3c, weights['L4a']), biases['L4a'])
    if batch_normalize:
        L4a = tf.nn.relu(Batch_Norm(L4a, is_training = is_training))
        L4b = tf.nn.bias_add(conv_valid(L4a, weights['L4b']), biases['L4b'])
        crop_4 = L4b[:, :, :, :]
        L4b = tf.nn.relu(Batch_Norm(L4b, is_training = is_training))
    else:
        L4a = tf.nn.relu(L4a)
        L4b = tf.nn.relu(tf.nn.bias_add(conv_valid(L4a, weights['L4b']), biases['L4b']))
        crop_4 = L4b[:, :, :, :]
    L4b = scSE(L4b, name = 'L4')
    L4c = pool(L4b)
    L4c = tf.nn.dropout(L4c, keep_prob)
    
    ## Block of layers number 5:
    L5a = tf.nn.bias_add(conv_valid(L4c, weights['L5a']), biases['L5a'])
    if batch_normalize:
        L5a = tf.nn.relu(Batch_Norm(L5a, is_training = is_training))
        L5b = tf.nn.bias_add(conv_valid(L5a, weights['L5b']), biases['L5b'])
        L5b = tf.nn.relu(Batch_Norm(L5b, is_training = is_training))
    else:
        L5a = tf.nn.relu(L5a)
        L5b = tf.nn.relu(tf.nn.bias_add(conv_valid(L5a, weights['L5b']), biases['L5b']))
    L5b = scSE(L5b, name = 'L5')
    
    ## Block of layers number 6:
    L6a = tf.nn.bias_add(tf.nn.conv2d_transpose(L5b, weights['L6a'], [batch_size, 14, 14, initial_filters*16], [1, 2, 2, 1], "VALID"),
                         biases['L6a'])
    if batch_normalize:
        L6a = tf.nn.relu(Batch_Norm(L6a, is_training = is_training))
        L6b = tf.nn.bias_add(conv_valid(L6a, weights['L6b']), biases['L6b'])
        L6b = tf.nn.relu(Batch_Norm(L6b, is_training = is_training))
        L6c = tf.nn.bias_add(conv_valid(L6b, weights['L6c']), biases['L6c'])
        L6c = tf.nn.relu(Batch_Norm(L6c, is_training = is_training))
    else:  
        L6a = tf.nn.relu(L6a)
        L6b = tf.nn.relu(tf.nn.bias_add(conv_valid(L6a, weights['L6b']), biases['L6b']))
        L6c = tf.nn.relu(tf.nn.bias_add(conv_valid(L6b, weights['L6c']), biases['L6c']))
    L6c = scSE(L6c, name = 'L6')
    L6c = tf.nn.dropout(L6c, keep_prob)
    
    ## Block of layers number 7:
    L7a = tf.nn.bias_add(tf.nn.conv2d_transpose(L6c, weights['L7a'], [batch_size, 20, 20, initial_filters*8], [1, 2, 2, 1], "VALID"),
                         biases['L7a'])
    if batch_normalize:
        L7 = tf.concat([crop_4, L7a], 3)
        L7 = tf.nn.relu(Batch_Norm(L7, is_training = is_training))
        L7b = tf.nn.bias_add(conv_valid(L7, weights['L7b']), biases['L7b'])
        L7b = tf.nn.relu(Batch_Norm(L7b, is_training = is_training))
        L7c = tf.nn.bias_add(conv_valid(L7b, weights['L7c']), biases['L7c'])
        L7c = tf.nn.relu(Batch_Norm(L7c, is_training = is_training))
    else:
        L7a = tf.nn.relu(L7a)
        L7 = tf.concat([crop_4, L7a], 3)
        L7b = tf.nn.relu(tf.nn.bias_add(conv_valid(L7, weights['L7b']), biases['L7b']))
        L7c = tf.nn.relu(tf.nn.bias_add(conv_valid(L7b, weights['L7c']), biases['L7c']))
    L7c = scSE(L7c, name = 'L7')
    L7c = tf.nn.dropout(L7c, keep_prob)
    
    ## Block of layers number 8:
    L8a = tf.nn.bias_add(tf.nn.conv2d_transpose(L7c, weights['L8a'], [batch_size, 32, 32, initial_filters*4], [1, 2, 2, 1], "VALID"),
                         biases['L8a'])
    if batch_normalize:
        L8 = tf.concat([crop_3, L8a], 3)
        L8 = tf.nn.relu(Batch_Norm(L8, is_training = is_training))
        L8b = tf.nn.bias_add(conv_valid(L8, weights['L8b']), biases['L8b'])
        L8b = tf.nn.relu(Batch_Norm(L8b, is_training = is_training))
        L8c = tf.nn.bias_add(conv_valid(L8b, weights['L8c']), biases['L8c'])
        L8c = tf.nn.relu(Batch_Norm(L8c, is_training = is_training))
    else:
        L8a = tf.nn.relu(L8a)
        L8 = tf.concat([crop_3, L8a], 3)
        L8b = tf.nn.relu(tf.nn.bias_add(conv_valid(L8, weights['L8b']), biases['L8b']))
        L8c = tf.nn.relu(tf.nn.bias_add(conv_valid(L8b, weights['L8c']), biases['L8c']))
    L8c = scSE(L8c, name = 'L8')
    L8c = tf.nn.dropout(L8c, keep_prob)
    
    ## Block of layers number 9:
    L9a = tf.nn.bias_add(tf.nn.conv2d_transpose(L8c, weights['L9a'], [batch_size, 56, 56, initial_filters*2], [1, 2, 2, 1], "VALID"),
                         biases['L9a'])
    if batch_normalize:
        L9 = tf.concat([crop_2, L9a], 3)
        L9 = tf.nn.relu(Batch_Norm(L9, is_training = is_training))
        L9b = tf.nn.bias_add(conv_valid(L9, weights['L9b']), biases['L9b'])
        L9b = tf.nn.relu(Batch_Norm(L9b, is_training = is_training))
        L9c = tf.nn.bias_add(conv_valid(L9b, weights['L9c']), biases['L9c'])
        L9c = tf.nn.relu(Batch_Norm(L9c, is_training = is_training))
    else:
        L9a = tf.nn.relu(L9a)
        L9 = tf.concat([crop_2, L9a], 3)
        L9b = tf.nn.relu(tf.nn.bias_add(conv_valid(L9, weights['L9b']), biases['L9b']))
        L9c = tf.nn.relu(tf.nn.bias_add(conv_valid(L9b, weights['L9c']), biases['L9c']))
    L9c = scSE(L9c, name = 'L9')
    L9c = tf.nn.dropout(L9c, keep_prob)
        
    ## Block of layers number 10:
    L10a = tf.nn.bias_add(tf.nn.conv2d_transpose(L9c, weights['L10a'], [batch_size, 104, 104, initial_filters], [1, 2, 2, 1], "VALID"),
                          biases['L10a'])
    if batch_normalize:
        L10 = tf.concat([crop_1, L10a], 3)
        L10 = tf.nn.relu(Batch_Norm(L10, is_training = is_training))
        L10b = tf.nn.bias_add(conv_valid(L10, weights['L10b']), biases['L10b'])
        L10b = tf.nn.relu(Batch_Norm(L10b, is_training = is_training))
        L10b = scSE(L10b, name = 'L10')
        L10c = tf.nn.bias_add(conv_valid(L10b, weights['L10c']), biases['L10c'])
        L10c = tf.nn.relu(Batch_Norm(L10c, is_training = is_training))
        L10d = tf.nn.bias_add(conv_valid(L10c, weights['L10d']), biases['L10d'])
        
    else:
        L10a = tf.nn.relu(L10a)
        L10 = tf.concat([crop_1, L10a], 3)
        L10b = tf.nn.relu(tf.nn.bias_add(conv_valid(L10, weights['L10b']), biases['L10b']))
        L10b = scSE(L10b, name = 'L10')
        L10c = tf.nn.bias_add(conv_valid(L10b, weights['L10c']), biases['L10c'])
        L10d = tf.nn.bias_add(conv_valid(L10c, weights['L10d']), biases['L10d'])
    return L10d
    
def Forward_UNet2(x, is_training, batch_size, decay, initial_filters = 64, apply_dropout = False, keep_prob = 0.5, batch_normalize = False):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    weights = {}
    biases = {}
    weights['L1a'] = tf.Variable(initializer([3, 3, 1, initial_filters]), name = 'wL1a')
    weights['L2a'] = tf.Variable(initializer([3, 3, initial_filters, initial_filters*2]), name = 'wL2a')
    weights['L3a'] = tf.Variable(initializer([3, 3, initial_filters*2, initial_filters*4]), name = 'wL3a')
    weights['L4a'] = tf.Variable(initializer([3, 3, initial_filters*4, initial_filters*8]), name = 'wL4a')
    weights['L5a'] = tf.Variable(initializer([3, 3, initial_filters*8, initial_filters*16]), name = 'wL5a')
    weights['L5b'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*16]), name = 'wL5b')
    weights['L5c'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*16]), name = 'wL5c')  
    weights['L6a'] = tf.Variable(initializer([2, 2, initial_filters*8, initial_filters*16]), name = 'wL6a')
    weights['L6b'] = tf.Variable(initializer([3, 3, initial_filters*16, initial_filters*8]), name = 'wL6b')
    weights['L7a'] = tf.Variable(initializer([2, 2, initial_filters*4, initial_filters*8]), name = 'wL7a')
    weights['L8a'] = tf.Variable(initializer([2, 2, initial_filters*2, initial_filters*8]), name = 'wL8a')
    weights['L8b'] = tf.Variable(initializer([3, 3, initial_filters*4, initial_filters*2]), name = 'wL8b')
    weights['L9a'] = tf.Variable(initializer([2, 2, initial_filters*1, initial_filters*2]), name = 'wL9a')
    weights['L9c'] = tf.Variable(initializer([6, 6, int(initial_filters*2), int(initial_filters*2)]), name = 'wL9b')
    weights['L9d'] = tf.Variable(initializer([3, 3, int(initial_filters*2), int(initial_filters)]), name = 'wL9c')
    weights['L9e'] = tf.Variable(initializer([1, 1, int(initial_filters), 1]), name = 'wL9d')
    
    biases['L1a'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL1a')
    biases['L2a'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL2a')
    biases['L3a'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL3a')
    biases['L4a'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL4a')
    biases['L5a'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL5a')
    biases['L5b'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL5b')  
    biases['L5c'] = tf.Variable(tf.zeros([initial_filters*16]), name = 'bL5c')  
    biases['L6a'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL6a')
    biases['L6b'] = tf.Variable(tf.zeros([initial_filters*8]), name = 'bL6b')
    biases['L7a'] = tf.Variable(tf.zeros([initial_filters*4]), name = 'bL7a')
    biases['L8a'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL8a')
    biases['L8b'] = tf.Variable(tf.zeros([initial_filters*2]), name = 'bL8b')
    biases['L9a'] = tf.Variable(tf.zeros([initial_filters]), name = 'bL9a')
    biases['L9c'] = tf.Variable(tf.zeros([int(initial_filters*2)]), name = 'bL9b')
    biases['L9d'] = tf.Variable(tf.zeros([int(initial_filters)]), name = 'bL9c')
    biases['L9e'] = tf.Variable(tf.zeros([1]), name = 'bL9d')
    
    ## Block of layers number 1:
    L1a = tf.nn.bias_add(conv_valid(x, weights['L1a']), biases['L1a'])
    crop_1 = L1a[:, 57: 57+108, 57: 57+108, :]
    L1b = Batch_Norm(L1a, is_training = is_training, decay = decay)
    L1b = tf.nn.dropout(L1b, keep_prob = keep_prob)
    L1b = tf.nn.relu(L1b)
    L1b = scSE(L1b, name = 'L1')
    L1c = pool(L1b)
#    L1c = tf.nn.dropout(L1c, keep_prob = keep_prob)

    ## Block of layers number 2:
    L2a = tf.nn.bias_add(conv_valid(L1c, weights['L2a']), biases['L2a'])
    crop_2 = L2a[:, 26: 26+56, 26: 26+56, :]
    L2a = Batch_Norm(L2a, is_training = is_training, decay = decay)
    L2a = tf.nn.dropout(L2a, keep_prob = keep_prob)
    L2a = tf.nn.relu(L2a)  
    L2b = scSE(L2a, name = 'L2')
    L2c = pool(L2b)

    ## Block of layers number 3:
    L3a = tf.nn.bias_add(conv_valid(L2c, weights['L3a']), biases['L3a'])
    crop_3 = L3a[:, 8: 8+28, 8: 8+28, :]
    L3a = Batch_Norm(L3a, is_training = is_training, decay = decay)
    L3a = tf.nn.dropout(L3a, keep_prob = keep_prob)
    L3a = tf.nn.relu(L3a)
    L3b = scSE(L3a, name = 'L3')
    L3c = pool(L3b)

    ## Block of layers number 4:
    L4a = tf.nn.bias_add(conv_valid(L3c, weights['L4a']), biases['L4a'])
    crop_4 = L4a[:, 4:4+16, 4:4+16, :]

    L4a = Batch_Norm(L4a, is_training = is_training, decay = decay)
    L4a = tf.nn.dropout(L4a, keep_prob = keep_prob)
    L4a = tf.nn.relu(L4a)
    L4b = scSE(L4a, name = 'L4')
    L4c = pool(L4b)

    ## Block of layers number 5:
    L5a = tf.nn.bias_add(conv_valid(L4c, weights['L5a']), biases['L5a'])
    L5a = Batch_Norm(L5a, is_training = is_training, decay = decay)
    L5a = tf.nn.dropout(L5a, keep_prob = keep_prob)
    L5a = tf.nn.relu(L5a)
    
    L5b = tf.nn.bias_add(conv_valid(L5a, weights['L5b']), biases['L5b'])
    L5b = Batch_Norm(L5b, is_training = is_training, decay = decay)
    L5b = tf.nn.dropout(L5b, keep_prob = keep_prob)
    L5b = tf.nn.relu(L5b)
    
    L5c = scSE(L5b, name = 'L5')
    
    ## Block of layers number 6:
    L6a = tf.nn.bias_add(tf.nn.conv2d_transpose(L5c, weights['L6a'], [batch_size, 16, 16, initial_filters*8], [1, 2, 2, 1], "VALID"),
                         biases['L6a'])
    L6 = tf.concat([crop_4, L6a], 3)
    L6a = Batch_Norm(L6, is_training = is_training, decay = decay)
    L6a = tf.nn.dropout(L6a, keep_prob = keep_prob)
    L6a = tf.nn.relu(L6a)
    
    L6b = tf.nn.bias_add(conv_valid(L6a, weights['L6b']), biases['L6b'])
    L6b = Batch_Norm(L6b, is_training = is_training, decay = decay)
    L6b = tf.nn.dropout(L6b, keep_prob = keep_prob)
    L6b = tf.nn.relu(L6b)
    L6c = scSE(L6b, name = 'L6')

    ## Block of layers number 7:
    L7a = tf.nn.bias_add(tf.nn.conv2d_transpose(L6c, weights['L7a'], [batch_size, 28, 28, initial_filters*4], [1, 2, 2, 1], "VALID"),
                         biases['L7a'])

    L7 = tf.concat([crop_3, L7a], 3)
    L7 = Batch_Norm(L7, is_training = is_training, decay = decay)
    L7 = tf.nn.dropout(L7, keep_prob = keep_prob)
    L7 = tf.nn.relu(L7)
    L7c = scSE(L7, name = 'L7')

    ## Block of layers number 8:
    L8a = tf.nn.bias_add(tf.nn.conv2d_transpose(L7c, weights['L8a'], [batch_size, 56, 56, initial_filters*2], [1, 2, 2, 1], "VALID"),
                         biases['L8a'])

    L8 = tf.concat([crop_2, L8a], 3)
    L8 = Batch_Norm(L8, is_training = is_training, decay = decay)
    L8 = tf.nn.dropout(L8, keep_prob = keep_prob)
    L8 = tf.nn.relu(L8)
    
    L8b = tf.nn.bias_add(conv_valid(L8, weights['L8b']), biases['L8b'])
    L8b = Batch_Norm(L8b, is_training = is_training, decay = decay)
    L8b = tf.nn.dropout(L8b, keep_prob = keep_prob)
    L8b = tf.nn.relu(L8b)
    
    L8c = scSE(L8b, name = 'L8')
    
    ## Block of layers number 9:
    L9a = tf.nn.bias_add(tf.nn.conv2d_transpose(L8c, weights['L9a'], [batch_size, 108, 108, initial_filters], [1, 2, 2, 1], "VALID"),
                         biases['L9a'])

    L9 = tf.concat([crop_1, L9a], 3)
    L9 = Batch_Norm(L9, is_training = is_training, decay = decay)
    L9 = tf.nn.dropout(L9, keep_prob = keep_prob)
    L9 = tf.nn.relu(L9)
    
    L9c = tf.nn.bias_add(conv_valid(L9, weights['L9c']), biases['L9c'])
    L9c = Batch_Norm(L9c, is_training = is_training, decay = decay)
    L9c = tf.nn.dropout(L9c, keep_prob = keep_prob)
    L9c = tf.nn.relu(L9c)
    
    L9c = scSE(L9c, name = 'L10')
    
    L9d = tf.nn.bias_add(conv_valid(L9c, weights['L9d']), biases['L9d'])
    L9d = Batch_Norm(L9d, is_training = is_training, decay = decay)
    L9d = tf.nn.dropout(L9d, keep_prob = keep_prob)
    L9d = tf.nn.relu(L9d)
    
    L9e = tf.nn.bias_add(conv_valid(L9d, weights['L9e']), biases['L9e'])   
        
    return L9e

def pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

def conv(a, w):
    return tf.nn.conv2d(a, w, strides = [1, 1, 1, 1], padding = 'SAME')

def conv_valid(a, w):
    return tf.nn.conv2d(a, w, strides = [1, 1, 1, 1], padding = 'VALID')

def Batch_Norm(x, is_training, decay = 0.99, scale = False, zero_debias = False):
    return tf.contrib.layers.batch_norm(x,
                                        decay = decay, 
                                        scale = scale, 
                                        is_training = is_training, 
                                        zero_debias_moving_mean = zero_debias) 
#                                        )
#    return tf.layers.batch_normalization(x,
#                                         axis=-1,
#                                         momentum=decay,
#                                         epsilon=0.001,
#                                         scale=scale,
#                                         beta_initializer=tf.zeros_initializer(),
#                                         gamma_initializer=tf.ones_initializer(),
#                                         moving_mean_initializer=tf.zeros_initializer(),
#                                         moving_variance_initializer=tf.ones_initializer(),
#                                         training=is_training,
#                                         trainable=True,
#                                         )
