import numpy as np

def softmax(s):
    return (np.exp(s) / (1 + np.exp(s)))

dat = np.genfromtxt('./hw3_train.dat', dtype=float)

data = dat[:, 0:10]
y = np.expand_dims(dat[:, 10], axis=1)
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

xi = np.linalg.inv(np.dot(np.transpose(data), data))
w = np.dot(np.dot(xi, np.transpose(data)), y)

y_head = np.dot(data, w)
lin_err = np.sum(np.power((y_head - y), 2)) / dat.shape[0]

print(lin_err.shape)

lr = 0.001
times = []

for i in range(1000):
    w = np.zeros((11, 1))
    iter = 0
    while(1):
         sample = np.random.randint(0, dat.shape[0])
         y_head = np.dot(data, w)
         print(y_head.shape)
         
         if np.sum((np.power((y_head - y), 2))) / dat.shape[0] <= 1.01 * lin_err:
             break
         w += lr * 2 * (y[sample, 0] - np.dot(data[sample, :], w)) * np.expand_dims(data[sample, :], axis=1)#softmax(-y[sample, 0] * y_head) * y[sample, 0] * np.expand_dims(data[sample, :], axis=1)
         iter += 1
    #print(iter)
    times.append(iter)

print(np.sum(times) / 1000)
