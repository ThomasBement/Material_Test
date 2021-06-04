# ---------------------------------------- #
# Time_Serries [Python File]
# Written By: Thomas Bement
# Created On: 2021-06-03
# ---------------------------------------- #

import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function for reading data into a dictonary
def readDict(location):
    R = open(location, 'r')
    ans = {}
    R.seek(0)
    Headers = R.readline().split(',')
    Headers[-1] = Headers[-1].replace('\n', '')
    for header in Headers:
        if (header != 'Local_Date_Time'):
            ans[header] = []
    R.seek(0)
    count = 0
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(',')
        for i in range(len(Headers)):
            if (Headers[i] != 'Local_Date_Time'):
                ans[Headers[i]].append(float(lineLis[i]))
        count += 1    
    for key in ans:
        ans[key] = np.array(ans[key])
    return ans

datPMS = readDict('.\Data\CSV\PMS_2021_06_01-13_05_09-TEST_002.csv')
datSEN = readDict('.\Data\CSV\SEN_2021-06-01_13-06-18-SPS3x_44B54F729AE1445F.csv')
datTSI = readDict('.\Data\CSV\TSI_2021_06_01-13_05_09-TEST_002.csv')

plt.plot(datPMS['Epoch_UTC'], datPMS['0.5_Bin'], color = '#f5ad42', alpha = 0.6)
plt.plot(datSEN['Epoch_UTC'], datSEN['NumbConc_0.5'], color = '#4287f5', alpha = 0.6)
plt.plot(datTSI['Epoch_UTC'], datTSI['0.465_Bin'], color = '#ad42f5', alpha = 0.6)

plt.plot(datPMS['Epoch_UTC'], datPMS['1.0_Bin'], color = '#f5ad42', alpha = 0.6)
plt.plot(datPMS['Epoch_UTC'], datPMS['2.5_Bin'], color = '#f5ad42', alpha = 0.6)


plt.plot(datSEN['Epoch_UTC'], datSEN['NumbConc_1.0'], color = '#4287f5', alpha = 0.6)
plt.plot(datSEN['Epoch_UTC'], datSEN['NumbConc_2.5'], color = '#4287f5', alpha = 0.6)


plt.plot(datTSI['Epoch_UTC'], datTSI['1.117_Bin'], color = '#ad42f5', alpha = 0.6)
plt.plot(datTSI['Epoch_UTC'], datTSI['2.685_Bin'], color = '#ad42f5', alpha = 0.6)

plt.legend(['PMS', 'SEN', 'TSI'])

plt.show()
plt.close()