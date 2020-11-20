import numpy as np

dat = np.genfromtxt('./hw3_train.dat', dtype=float)
print(dat.shape)

data = dat[:, 0:10]
y = np.expand_dims(dat[:, 10], axis=1)
print(data.shape)
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

xi = np.linalg.inv(np.dot(np.transpose(data), data))
w = np.dot(np.dot(xi, np.transpose(data)), y)

y_head = np.dot(data, w)
err = np.sum(np.power((y_head - y), 2)) / y.shape[0]
print(err)
