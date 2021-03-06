import numpy as np
from random import choice

size = 20
result = 0

def h(x, theta, s): 
    y = x.copy()
    y[(y - theta) > 0] = 1
    y[(y - theta) <= 0] = -1
    y *= s
    return y

def fit(x, y):
    theta = np.append(-1, (x[1:] + x[:-1]) / 2)
    train_x = np.repeat(np.expand_dims(x, axis=0), 2 * len(theta), axis=0)
    l = [[-1], [1]]
    s = np.repeat(np.tile(l, (len(theta), 1)), len(x), axis=1)
    theta_ = np.repeat(theta, 2, axis=0)
    theta_ = np.repeat(np.expand_dims(theta_, axis=0), len(x), axis=1).reshape(train_x.shape)
    
    #print(train_x)
    #print(theta_)
    #print(s)
    y_head = h(train_x, theta_, s)
    #print(np.array(y))
    #print(np.array(y_head))
    acc = np.sum((y_head == y).astype(np.int), axis=1) / len(y)
    #print(acc)
    #print(np.argmax(acc) // 2)
    #print(np.argmax(acc) % 2)
    return theta[np.argmax(acc) // 2], l[np.argmax(acc) % 2], acc.max()


for i in range(10000):
    x_in = np.random.uniform(-1, 1 + 1e-10, size)
    x_in.sort()
    y_in = h(x_in, 0, 1)
    theta, s ,acc = fit(x_in, y_in)
    
    x_out = np.random.uniform(-1, 1 + 1e-10, 100000)
    x_out.sort()
    y_out = h(x_out, 0, 1)
    y_predict = h(x_out, theta, s)
    
    test_acc = np.sum(y_out == y_predict).astype(np.int) / len(x_out)
    result += ((1 - test_acc) - (1 - acc))



print(result / 10000)
