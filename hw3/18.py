import numpy as np

def sign(x):
    y = x.copy()
    y[y > 0] = 1
    y[y <= 0] = -1
    return y 

dat = np.genfromtxt('./hw3_train.dat', dtype=float)
t = np.genfromtxt('./hw3_test.dat', dtype=float)


data = dat[:, 0:10]
y_d = np.expand_dims(dat[:, 10], axis=1)
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

xi = np.linalg.inv(np.dot(np.transpose(data), data))
w = np.dot(np.dot(xi, np.transpose(data)), y_d)

y_head = np.dot(data, w)
err_in = np.sum(sign(y_head) != y_d) / data.shape[0]

test = t[:, 0:10]
y_t = np.expand_dims(t[:, 10], axis=1)
test = np.concatenate((np.ones((test.shape[0], 1)), test), axis=1)
y_test = np.dot(test, w)
err_out = np.sum(sign(y_test) != y_t) / test.shape[0]

print(err_in - err_out)
