import numpy as np

def sigmoid(s):
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
    for j in range(500):
         sample = np.random.randint(0, dat.shape[0])
         y_head = np.dot(w.T,data[sample, :])
         
         #if np.power((y_head - y[sample, 0]), 2) <= 1.01 * lin_err:
         #    break
         w += lr * sigmoid(-y[sample, 0] * y_head) * y[sample, 0] * np.expand_dims(data[sample, :], axis=1)
         
    err = np.power((sigmoid(np.dot(data, w)) - y), 2) / dat.shape[0]  
    times.append(err)

print(np.sum(times) / 1000)
