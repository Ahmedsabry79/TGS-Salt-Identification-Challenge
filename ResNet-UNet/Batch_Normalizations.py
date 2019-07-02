# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:37:30 2019

@author: ASabry
"""

import tensorflow as tf


def Batch_Normalization(x, name, is_training,  decay = 0.99):
    if is_training is not None:
        shape = x.get_shape().as_list()
        pop_mean = tf.Variable(tf.zeros(shape = shape[-1]),trainable = False, name = 'pop_mean'+name)
        pop_var = tf.Variable(tf.ones(shape = shape[-1]), trainable = False, name = 'pop_var'+name)
        
        gamma = tf.Variable(tf.ones([shape[-1]]), name = 'gamme'+name)
        beta = tf.Variable(tf.zeros([shape[-1]]), name = 'beta'+name)
        
        batch_mean, batch_var = tf.nn.moments(x, axes = [0, 1, 2])
        train_mean = tf.assign(pop_mean, decay*pop_mean+(1-decay)*batch_mean, name = 'Pop_Mean'+name)
        train_var = tf.assign(pop_var, decay*pop_var+(1-decay)*batch_var, name = 'Pop_Var'+name)
        mean, var = tf.cond(is_training, lambda:(batch_mean, batch_var), lambda: (train_mean, train_var))
        return tf.nn.batch_normalization(x, mean, var, offset = beta, scale = gamma, variance_epsilon = 0.001)
        
def Batch_Normalization_No_Scale(x, name, is_training,  decay = 0.99):
    if is_training is not None:
        shape = x.get_shape().as_list()
        pop_mean = tf.Variable(tf.zeros(shape = shape[-1]),trainable = False, name = 'pop_mean'+name)
        pop_var = tf.Variable(tf.ones(shape = shape[-1]), trainable = False, name = 'pop_var'+name)
        batch_mean, batch_var = tf.nn.moments(x, axes = [0, 1, 2])
        train_mean = tf.assign(pop_mean, decay*pop_mean+(1-decay)*batch_mean, name = 'Pop_Mean'+name)
        train_var = tf.assign(pop_var, decay*pop_var+(1-decay)*batch_var, name = 'Pop_Var'+name)
        mean, var = tf.cond(is_training, lambda:(batch_mean, batch_var), lambda: (train_mean, train_var))
        return tf.nn.batch_normalization(x, mean, var, offset = None, scale = None, variance_epsilon = 0.001)
    
def Batch_Normalization1(x, name, decay = 0.999, is_training = True):
    shape = x.get_shape().as_list()
    pop_mean = tf.Variable(tf.zeros(shape = shape[-1]),trainable = False, name = 'pop_mean'+name)
    pop_var = tf.Variable(tf.ones(shape = shape[-1]), trainable = False, name = 'pop_var'+name)
    
    gamma = tf.Variable(tf.ones([shape[-1]]), name = 'gamme'+name)
    beta = tf.Variable(tf.zeros([shape[-1]]), name = 'beta'+name)
    
    batch_mean, batch_var = tf.nn.moments(x, axes = [0, 1, 2])
    
    if is_training:
        return tf.nn.batch_normalization(x, batch_mean, batch_var, offset = beta, scale = gamma, variance_epsilon = 0.001)
    else:
        train_mean = tf.assign(pop_mean, decay*pop_mean+(1-decay)*batch_mean)
        train_var = tf.assign(pop_var, decay*pop_var+(1-decay)*batch_var)  
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset = beta, scale = gamma, variance_epsilon = 0.001)
        


def Batch_Normalization2(x, decay = 0.999, is_training = True):
    shape = x.get_shape().as_list()
    pop_mean = tf.Variable(tf.zeros(shape = shape[-1]),trainable = False)
    pop_var = tf.Variable(tf.ones(shape = shape[-1]), trainable = False)
    
    gamma = tf.Variable(tf.ones([shape[-1]]))
    beta = tf.Variable(tf.zeros([shape[-1]]))
    
    batch_mean, batch_var = tf.nn.moments(x, axes = [0, 1, 2])
    
    if is_training:
        train_mean = tf.assign(pop_mean, decay*pop_mean+(1-decay)*batch_mean)
        train_var = tf.assign(pop_var, decay*pop_var+(1-decay)*batch_var)
        return tf.nn.batch_normalization(x, batch_mean, batch_var, offset = beta, scale = gamma, variance_epsilon = 0.001)
    else:
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, train_mean, train_var, offset = beta, scale = gamma, variance_epsilon = 0.001)



def Batch_Normalization_(x, decay = 0.999, is_training = True):
    shape = x.get_shape().as_list()
    
    batch_mean, batch_var = tf.nn.moments(x, axes = [0, 1, 2])
    
    gamma = tf.Variable(tf.ones([shape[-1]]))
    beta = tf.Variable(tf.zeros([shape[-1]]))
    
    ema = tf.train.ExponentialMovingAverage(decay = decay)
    ema_op = ema.apply([batch_mean, batch_var])
    
    if is_training:
        return tf.nn.batch_normalization(x, batch_mean, batch_var, offset = beta, scale = gamma, variance_epsilon = 0.001)
    else:
        with tf.control_dependencies([ema_op]):
            pop_mean = ema.average(batch_mean)
            pop_var = ema.average(batch_var)
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset = beta, scale = gamma, variance_epsilon = 0.001), ema_op


def Batch_Normalization_3(x, decay = 0.999, is_training = True):
    shape = x.get_shape().as_list()
    
    batch_mean, batch_var = tf.nn.moments(x, axes = [0, 1, 2])
    
    gamma = tf.Variable(tf.ones([shape[-1]]))
    beta = tf.Variable(tf.zeros([shape[-1]]))
    
    ema = tf.train.ExponentialMovingAverage(decay = decay, zero_debias = False)
    ema_op = ema.apply([batch_mean, batch_var])
    pop_mean = ema.average(batch_mean)
    pop_var = ema.average(batch_var)

    if is_training:
        return tf.nn.batch_normalization(x, batch_mean, batch_var, offset = beta, scale = gamma, variance_epsilon = 0.001)
    else:
        return tf.nn.batch_normalization(x, pop_mean, pop_var, offset = beta, scale = gamma, variance_epsilon = 0.001), ema_op







