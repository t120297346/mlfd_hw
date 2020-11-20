import numpy as np

def sigmoid(s):
    return (np.exp(s) / (1 + np.exp(s)))

dat = np.genfromtxt('./hw3_train.dat', dtype=float)

data = dat[:, 0:10]
y = np.expand_dims(dat[:, 10], axis=1)
data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

xi = np.linalg.inv(np.dot(np.transpose(data), data))
w_lin = np.dot(np.dot(xi, np.transpose(data)), y)

y_head = np.dot(data, w_lin)
lin_err = np.sum(np.power((y_head - y), 2)) / dat.shape[0]

print(lin_err.shape)

lr = 0.001
times = []

for i in range(1000):
    w = w_lin
    for j in range(500):
         sample = np.random.randint(0, dat.shape[0])
         y_head = np.dot(data[sample, :], w)
         
         w += lr * sigmoid(-y[sample, 0] * y_head) * y[sample, 0] * np.expand_dims(data[sample, :], axis=1)
         
    err = np.log((1 + np.exp(-y * np.dot(data, w)))) / dat.shape[0]  
    times.append(err)

print(np.sum(times) / 1000)
