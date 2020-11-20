import random
import numpy as np

def sign(num):
    return 1 if num > 0 else -1

dat = np.genfromtxt('./hw1_train.dat', dtype=float)
data = np.concatenate((dat[:, :10], np.ones((dat.shape[0], 1))), axis=1)
w0 = []

for i in range(1000):
    weight = np.zeros((11,1))
    count = 0
    update = 0
    while(1):
         sample = random.randint(0, dat.shape[0] - 1)
         if (dat[sample, 10] * sign(np.expand_dims(data[sample, :], axis = 0) @ weight)) < 0:
             weight +=  (dat[sample, 10] * np.expand_dims(data[sample, :], axis = 0)).T
             count = 0
             update += 1
         else:
             count += 1
         if count == 500:
             break
    w0.append(weight[10, 0])
w0 = np.sort(w0)
print(w0[499:501])    
