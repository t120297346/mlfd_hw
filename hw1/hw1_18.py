import numpy as np

def sign(num):
    return 1 if num > 0 else -1

dat = np.genfromtxt('./hw1_train.dat', dtype=float)
update_time = []

for i in range(1000):
    weight = np.zeros((11,1))
    count = 0
    update = 0
    while(1):
         sample = np.random.randint(0, dat.shape[0])
         data = dat[sample, 0:10]
         data = np.append(data, [10])
         data = data.reshape((11,1))
         if dat[sample, 10] * sign(np.transpose(data) @ weight) < 0:
             weight = np.add(weight, dat[sample, 10] * data)
             count = 0
             update += 1
         else:
             count += 1
         if count == 500:
             break
    update_time.append(update)
update_time = np.sort(update_time, axis=None)
print(update_time)
print(update_time[499:501])    
