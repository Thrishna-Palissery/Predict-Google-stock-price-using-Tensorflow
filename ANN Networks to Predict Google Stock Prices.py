# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:48:19 2017

@author: Thrishna
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
output = ['Mostly cloudy', 'Sunny', 'rainy']
training_set = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:,1:2].values
sc = MinMaxScaler() 
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]

X = tf.placeholder(tf.float32, [None, 1], name='feature')
Y = tf.placeholder(tf.float32, [None,1], name='label')
w1 = tf.Variable(tf.random_normal(shape=[1,1], stddev=0.01), name='weights1')
b1 = tf.Variable(tf.random_normal([1]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, w1) + b1)
w2 = tf.Variable(tf.random_normal(shape=[1, 1], stddev=0.01), name='weights2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
logits = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)

cost = tf.reduce_mean(tf.squared_difference(logits, Y))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values 
testing_set = sc.fit_transform(real_stock_price)
test_Y = testing_set[1:20]
X_test = testing_set[0:19]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    for step in range(1000): 
        cost_val, _ = sess.run([cost, train], feed_dict={X: X_train, Y: y_train})
        if step % 200 == 0: 
            print(step, cost_val)
    test_predict = sess.run(logits, feed_dict={X: X_test})
    predicted_stock_price = sc.inverse_transform(test_predict)
plt.figure(2) 
plt.plot(real_stock_price, color ="red", label ="Real")
plt.plot(predicted_stock_price, color ="blue", label ="Prediction")
plt.legend(loc='upper left', frameon=False) 
plt.show()