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
#        init = tf.initialize_all_variables()
        
        
    def _create_network(self):
        self.w = tf.Variable(0.0, name="weights")
        self.output = tf.mul(self.x, self.w)
        self.saver = tf.train.Saver()
        
    def _create_loss_optimizer(self):
        self.cost = tf.square(self.y - self.output)
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)
    
    def fit(self, x,y):
        
#        self.saver = tf.train.Saver()
        opt, cost= self.sess.run((self.optimizer, self.cost), \
                                  feed_dict={self.x:x,self.y:y})
                                  
#    def save(self, checkpoint_dir, step):
#        model_name = "LinearReg.model"
##        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
##        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
#        model_dir = '/model'
#
#        if not os.path.exists(checkpoint_dir):
#            os.makedirs(checkpoint_dir)
#
#        self.saver.save(self.sess,
#                        os.path.join(model_dir, model_name),
#                        global_step=step)
#
#    #LOAD MODEL
#    def load(self, checkpoint_dir):
#        print(" [*] Reading checkpoints...")
#
##        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
##        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
#        model_dir = '/model'
#        ckpt = tf.train.get_checkpoint_state(model_dir)
#        if ckpt and ckpt.model_checkpoint_path:
#            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
#            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
#            return True
#        else:
#            return False