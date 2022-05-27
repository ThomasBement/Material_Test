# ---------------------------------------- #
# OverlayPlot [Python File]
# Written By: Thomas Bement
# Created On: 2021-08-10
# ---------------------------------------- #

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

fil_name = 'Overlay_Master_Data.csv'
path_name = '.\Data\CSV'

df = pd.read_csv('%s\%s' %(path_name, fil_name))
print(df)

chanel = '11'
plt_dat = [[], [], []]
for i in range(len(df)):
    if (df['Structure'][i] == 'Woven'):
        plt_dat[-1].append('%s, %s' %(df['Simple name'][i], df['Sample code'][i]))
        plt_dat[0].append(df['Pressure drop [Pa]'][i])
        plt_dat[1].append(1 - df[chanel][i])

for i in range(len(plt_dat[0])):
    plt.scatter(plt_dat[0][i], plt_dat[1][i])
plt.legend(plt_dat[2])
plt.show()
plt.close()