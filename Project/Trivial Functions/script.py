import subprocess
import os
import numpy as np

functions = ['B1', 'B2', 'B3', 'BL', 'RB', 'ST']
optims = ['GD', 'CM', 'Adam', 'Adagrad', 'Adadelta', 'RMSprop']

for i in functions:
    for j in optims:
        subprocess.call(["python3", "main.py", "--optim", j, "--fn", i]) 

for directory_1 in functions:
    for directory_2 in optims:
        data = []
        file_list = os.listdir(directory_1+'/'+directory_2)
        for filename in file_list:
            filepath = directory_1 +'/'+ directory_2 +'/'+ filename
            data.append(np.genfromtxt(filepath)[-1,1])
        print(directory_1, directory_2, file_list[np.argsort(data)[-1]])
