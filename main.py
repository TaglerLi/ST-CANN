import numpy as np
import os
import pandas as pd
from CANN import CANN
import numpy as np
import pandas as pd
from jpgtomp4 import visulize
from matplotlib import pyplot as plt

path = 'F:\Desktop\Dataset (1)\Dataset\game9\Clip3'
datafile = path + '\Label.csv'
data = pd.read_csv(datafile).to_numpy()

nx = CANN(1280)
ny = CANN(720)
figurename = []
locationx = []
locationy = []
tempx = -1
tempy = -1
prex = []
prey = []
index = 0

for item in data:
    print(index,len(data))
    index += 1
    if item[1] != 0:
        figurename.append(item[0])
        locationx.append(item[2])
        locationy.append(item[3])
        deltax = np.linspace(tempx, item[2], 5)
        if tempx != -1:
            for slice in deltax:
                predictedx = nx.predict(slice)
            prex.append(int(predictedx))
        if tempy != -1:
            deltay = np.linspace(tempy, item[3], 70)
            for slice in deltay:
                predictedy = ny.predict(slice)
            prey.append(int(predictedy))
        tempx = item[2]
        tempy = item[3]

dataout = pd.DataFrame([prex,prey])
dataout.to_excel('pre.xlsx')
visulize(path)