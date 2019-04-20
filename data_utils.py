
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#defining time frame of 1s with steps of 5ms
T = 1
dt = 0.005
time  = np.arange(0, T+dt, dt)

# Sliding window and initializing weights
w = np.zeros([5,5])
m_pot = np.zeros([28,28])
r = [-2,-1,0,1,2]
w[2][2] = 1

for i in range(5):
    for j in range(5):
        d = abs(2-i) + abs(2-j)
        w[i][j] = (-0.375)*d + 1

def encode(img, spike = False):
    #generating membrane potential from pixels
    if(spike):
        img = (2*img) -1
        img = np.array(img).reshape((28,28))
        for i in range(28):
            for j in range(28):
                s = 0
                for p in r:
                    for q in r:
                        if (i+p)>=0 and (i+p)<=27 and (j+q)>=0 and (j+q)<=27:
                            s = s + w[2+p][2+q]*img[i+p][j+q]
                m_pot[i][j] = s
        #print(m_pot)
        #initializing spike train
        spike_train = []
        for i in range(28):
            for j in range(28):
                temp = np.zeros([len(time),])
                #calculating firing rate proportional to the membrane potential
                f = math.ceil(m_pot[i][j] + 52.02)
                f1 = math.ceil((len(time)-1)/f)
                
                #generating spikes according to the firing rate
                k = 0
                while k<len(time)-1:
                    temp[k] = 1
                    k = k + f1
                spike_train.append(temp)
        return spike_train
    else:
        return img

img1 = mnist.train.images[0]
#calling encode function to print membrane potential and spike train.

encode(img1) #got 784 scalar values, one for each pixel
# to get spike train: encode(img1,True)

#Note: The generated spike train/scalar value can be applied to learning methods





