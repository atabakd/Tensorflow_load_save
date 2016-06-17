# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:48:10 2016

@author: atabak
"""

import tensorflow as tf
from LinearRegression import linearRegression
import numpy as np
import os

with tf.Session() as sess:
    LR = linearRegression(sess)
    LR.saver.restore(LR.sess, "simple.ckpt")
    print(sess.run(LR.w))
    