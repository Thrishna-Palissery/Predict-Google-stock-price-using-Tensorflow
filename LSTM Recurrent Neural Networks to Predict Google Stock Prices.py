# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:05:54 2017

@author: Thrishna
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

timesteps = sequence_length = 1
data_dim = 1
hidden_dim = 5
output_dim = 1
learning_rate = .01
iterations = 600

training_set = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:,1:2].values
sc = MinMaxScaler() 
training_set = sc.fit_transform(training_set)

# Getting the inputs and the outputs, y_train is output, x_train is the input
X_train = training_set[0:1257]
y_train = training_set[1:1258]
X_train = np.reshape(X_train, (1257, 1, 1))

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values 
testing_set = sc.fit_transform(real_stock_price)
test_Y = testing_set[1:20]
X_test = testing_set[0:19]
X_test = np.reshape(X_test , (19, 1, 1)) 

X = tf.placeholder(tf.float32, (None, sequence_length,1))
Y = tf.placeholder(tf.float32, (None,1))
with tf.variable_scope('cel'):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=5,state_is_tuple=True)
with tf.variable_scope('rn'):
    outputs, _ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
with tf.variable_scope('full'):
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)

loss = tf.reduce_sum(tf.square(Y_pred - Y)) 

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    acc_t = 0
    for i in range(iterations):
            _, step_loss,acc = sess.run([train, loss,accuracy], feed_dict={
                                X: X_train, Y: y_train})
            print("[step: {}] loss: {}".format(i, step_loss))
            acc_t += acc

    # Test step
    print("ACCURACY:",(acc_t/iterations))
    train_predict = sess.run(Y_pred, feed_dict={X: X_train})
    predicted_train_stock_price = sc.inverse_transform(train_predict)
    y_train = sc.inverse_transform(y_train)
    test_predict = sess.run(Y_pred, feed_dict={X: X_test})
    predicted_stock_price = sc.inverse_transform(test_predict)
  

# Visualising the results
plt.figure(1) 
plt.title('Google Stock Price Prediction - Train Data')
plt.plot(y_train, color ="red", label ="Real")
plt.plot(predicted_train_stock_price, color ="blue", label ="Prediction")
plt.legend(loc='upper left', frameon=False) 
plt.show()

plt.figure(2) 
plt.title('Google Stock Price Prediction - Test Data')
plt.plot(real_stock_price, color ="red", label ="Real")
plt.plot(predicted_stock_price, color ="blue", label ="Prediction")
plt.legend(loc='upper left', frameon=False) 
plt.show()