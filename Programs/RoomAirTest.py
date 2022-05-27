# ---------------------------------------- #
# RoomAirTest [Python File]
# Written By: Thomas Bement
# Created On: 2021-07-27
# ---------------------------------------- #

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filName = 'compare_atomizer_07272021.csv'
pathName = '.\Data\CSV'
sampleSize = 5

df = pd.read_csv('%s\%s' %(pathName, filName))

allDat = {}
for key in df:
    allDat[key] = np.array(df[key])
plotDat = [[], [], [], [], []]
percentDat = [[], [], []]
for i in range(len(allDat['size'])):
    if ('.' in allDat['size'][i]):
        plotDat[0].append(allDat['size'][i])
        plotDat[1].append(allDat['bypass_avg'][i])
        plotDat[2].append(allDat['bypass_std'][i]/math.sqrt(sampleSize))
        plotDat[3].append(allDat['off_avg'][i])
        plotDat[4].append(allDat['off_std'][i]/math.sqrt(sampleSize))

        print('--------------------------------')
        print('Results for %s:' %(allDat['size'][i]))
        a = allDat['off_avg'][i]
        b = allDat['bypass_avg'][i]
        a_SE = allDat['off_std'][i]/math.sqrt(sampleSize)
        b_SE = allDat['bypass_std'][i]/math.sqrt(sampleSize)
        print('Nominal percentage of mesured results: %.4f %%' %(100*a/b))
        print('Associated STD: %.4f %%' %(math.sqrt( (100*a_SE/b)**2 + ((100*a*b_SE)/(b**2))**2 )))
        print('--------------------------------')

        percentDat[0].append(allDat['size'][i])
        percentDat[1].append(100*a/b)
        percentDat[2].append(math.sqrt( (100*a_SE/b)**2 + ((100*a*b_SE)/(b**2))**2 ))

print('--------------------------------')
print('Results for the %s:' %(allDat['size'][-1]))
a = allDat['off_avg'][-1]
b = allDat['bypass_avg'][-1]
a_SE = allDat['off_std'][-1]/math.sqrt(sampleSize)
b_SE = allDat['bypass_std'][-1]/math.sqrt(sampleSize)
print('Nominal percentage of mesured results: %.4f %%' %(100*a/b))
print('Associated STD: %.4f %%' %(math.sqrt( (100*a_SE/b)**2 + ((100*a*b_SE)/(b**2))**2 )))
print('--------------------------------')

plt.title('Comparison of Nominal Room Air Particle Counts and Atomizer Particle Counts')
plt.ylabel('Average Particle Counts [#]')
plt.xlabel('Particle Size [um]')
plt.xscale('log',base=10) 
plt.yscale('log',base=10) 
plt.xticks(rotation = 90)
plt.errorbar(plotDat[0], plotDat[1], yerr=plotDat[2], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
plt.errorbar(plotDat[0], plotDat[3], yerr=plotDat[4], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
plt.show()
plt.close()

plt.title('Nominal Percentage of Counts for Room Air')
plt.ylabel('Room/Atomizer [%]')
plt.xlabel('Particle Size [um]')
plt.xscale('log',base=10) 
plt.xticks(rotation = 90)
plt.errorbar(percentDat[0], percentDat[1], yerr=percentDat[2], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
plt.show()
plt.close()