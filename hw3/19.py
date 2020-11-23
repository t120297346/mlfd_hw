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



test = t[:, 0:10]
y_t = np.expand_dims(t[:, 10], axis=1)
test = np.concatenate((np.ones((test.shape[0], 1)), test), axis=1)

second = np.power(data[:, 1:], 2)
triple = np.power(data[:, 1:], 3)
phi = np.concatenate((data, second, triple), axis=1)

phi_i = np.linalg.inv(np.dot(np.transpose(phi), phi))
w = np.dot(np.dot(phi_i, phi.T), y_d)

y_head = np.dot(phi, w)
err_in = np.sum(sign(y_head) != y_d) / data.shape[0]

test_sec = np.power(test[:, 1:], 2)
test_tri = np.power(test[:, 1:], 3)
phi_t = np.concatenate((test, test_sec, test_tri), axis=1)

y_thead = np.dot(phi_t, w)
err_out = np.sum(sign(y_thead) != y_t) / test.shape[0]
print(err_in - err_out)

