# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:32:45 2016

@author: atabak
"""

import tensorflow as tf
from LinearRegression import linearRegression
import numpy as np
import os

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

with tf.Session() as sess:
    LR = linearRegression(sess)
    for i in range(100):
        for (x, y) in zip(trX, trY):
            LR.fit(x,y)
    print(sess.run(LR.w))
    path = LR.saver.save(LR.sess, os.path.join(os.path.dirname(__file__), "simple.ckpt"))
    print("Saved:", path)      

