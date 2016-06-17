# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:56:27 2016

@author: atabak
"""

import tensorflow as tf
#import os

class linearRegression(object):

    def __init__(self, sess):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.sess = sess
        
        self._create_network()     
        tf.initialize_all_variables().run()
        self._create_loss_optimizer()
        
    def _create_network(self):
        self.w = tf.Variable(0.0, name="weights")
        self.output = tf.mul(self.x, self.w)
        self.saver = tf.train.Saver()
        
    def _create_loss_optimizer(self):
        self.cost = tf.square(self.y - self.output)
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)
    
    def fit(self, x,y):
        opt, cost= self.sess.run((self.optimizer, self.cost), \
                                  feed_dict={self.x:x,self.y:y})
                                  