# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:52:33 2018

@author: onee
"""

import numpy as np
import tensorflow as tf

import os
from PIL import Image
import random

def T_CNN_3(x):
     #x=tf.placeholder(tf.float32, [None,227,227,3])
#    print(x.get_shape().as_list())
    
    x = tf.layers.conv2d(
            x,
            filters=96,
            kernel_size=11,
            strides=(4, 4),
            padding='valid',
            activation=tf.nn.relu)
    # Max pool
    x = tf.layers.max_pooling2d(
            x,
            pool_size=3,
            strides=2,
            padding='valid')
    
    
    x = tf.layers.conv2d(
            x,
            filters=256,
            kernel_size=5,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu)
    # Max pool
    x = tf.layers.max_pooling2d(
            x,
            pool_size=3,
            strides=2,
            padding='valid')
    
    
    x = tf.layers.conv2d(
            x,
            filters=384,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu)
    
    
    
    x = tf.nn.avg_pool(
            x,
            ksize=[1, 13, 13, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        
    x = tf.reshape(x, [-1, 1 * 1 * 384])
    
    
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
    x = tf.nn.dropout(x, 0.8)
    
    
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
    x = tf.nn.dropout(x, 0.8)
    
    
    x = tf.layers.dense(x, 2, activation=None)
        
    return x
        
        
def main(unused_args):
    train_list = []
    train_label = []
    train_zip = []
    
    # Load Training Data
    photo_data_dir = './data/photo/'
    painting_data_dir = './data/painting/'
    
    photo_file_list = os.listdir(photo_data_dir)
    painting_file_list = os.listdir(painting_data_dir)        

    for i in range(len(photo_file_list)):
        filenames = photo_data_dir + photo_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((3000, 3000), Image.ANTIALIAS)
        train_image = train_image.crop((1387,1387,1614,1614))
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(0)
        
    for i in range(len(painting_file_list)):
        filenames = painting_data_dir + painting_file_list[i]
        train_image = Image.open(filenames)
        train_image = train_image.convert("RGB")
        train_image = train_image.resize((3000, 3000), Image.ANTIALIAS)
        train_image = train_image.crop((1387,1387,1614,1614))
        train_image = np.array(train_image, dtype=np.float32)
        # train_list0.append(train_image) 
        train_list.append(train_image)
        
        train_label.append(1)
        
    train_zip = list(zip(train_list, train_label))
    random.shuffle(train_zip)
    
    train_list, train_label = zip(*train_zip)
    train_label = np.asarray(train_label) 
    train_list = np.asarray(train_list)
    
    #label one-hot encoding
    max_value = np.max(train_label) + 1
    train_label = np.eye(max_value)[train_label]
    
    
    
    #Load Test Data
    test_list = []
    test_label = []
    test_zip = []
    
    #Load Test Data
    photo_data_dir2 = './data/photo_test/'
    painting_data_dir2 = './data/painting_test/'
    
    photo_file_list2 = os.listdir(photo_data_dir2)
    painting_file_list2 = os.listdir(painting_data_dir2)        
        
    for i in range(len(photo_file_list2)):
        filenames = photo_data_dir2 + photo_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((3000, 3000), Image.ANTIALIAS)
        test_image = test_image.crop((1387,1387,1614,1614))
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(0)
        
    for i in range(len(painting_file_list2)):
        filenames = painting_data_dir2 + painting_file_list2[i]
        test_image = Image.open(filenames)
        test_image = test_image.convert("RGB")
        test_image = test_image.resize((3000, 3000), Image.ANTIALIAS)
        test_image = test_image.crop((1387,1387,1614,1614))
        test_image = np.array(test_image, dtype=np.float32)
        # train_list0.append(train_image) 
        test_list.append(test_image)
        
        test_label.append(1)
        
    
    test_zip = list(zip(test_list, test_label))
    random.shuffle(test_zip)
    
    test_list, test_label = zip(*test_zip)
    test_label = np.asarray(test_label) 
    test_list = np.asarray(test_list)
    
    #label one-hot encoding
    max_value = np.max(test_label) + 1
    test_label = np.eye(max_value)[test_label]
    
    ####################################################################################################
    
    X = tf.placeholder("float", [None, 227, 227, 3])
    Y = tf.placeholder("float", [None, 2])
    
    tcnn_x = T_CNN_3(X)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tcnn_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.0005).minimize(cost)
    predict_op = tf.argmax(tcnn_x, 1)
    
    
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()
        
        train_epoch = 10
        batch_size = 20
        
        for i in range(train_epoch):
            batch_count = int(len(train_list)/batch_size)
            for j in range(batch_count):
                batch_x = train_list[(i*batch_size):((i+1)*batch_size)]
                batch_y = train_label[(i*batch_size):((i+1)*batch_size)]
                sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
            print(i)
        
        print("test accuracy: ",np.mean(np.argmax(test_label, axis=1) ==
                                        sess.run(predict_op, feed_dict={X:test_list, Y:test_label})))
            
#        r_testing = sess.run(predict_op, feed_dict={X:test_list, Y:test_label})
#        print(r_testing, sess.run(tf.arg_max(r_testing,1)))        
        
if __name__ == '__main__':
    tf.app.run()    
