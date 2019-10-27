# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:29:40 2019

@author: mhossa12
"""

import tensorflow as tf
import numpy as np

c = np.array([[3.,4], [5.,6], [6.,7]])
step = tf.reduce_mean(c, 1)                                                                                 
with tf.compat.v1.Session() as sess:
    print(sess.run(step))