# ---------------------------------------- #
# DataAnalysis [Python File]
# Written By: Thomas Bement
# Created On: 2021-06-28
# ---------------------------------------- #
"""
IMPORTS
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Save starts file to dictonary, this function removes off by one error for the samples
def readStarts(locationStarts):
    ans = {'name': [], 'dp': [], 'sample': []}
    R = open(locationStarts, 'r')
    R.seek(0)
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(',')
        ans['name'].append(lineLis[0])
        ans['dp'].append(float(lineLis[1]))
        if '\n' in lineLis[2]:
            ans['sample'].append(int(lineLis[2].replace('\n', '')) - 1)
        else:
            ans['sample'].append(int(lineLis[2]) - 1)
    return ans

def readDat(location):
    ans = {}
    headers = []
    R = open(location, 'r')
    R.seek(0)
    for line in itertools.islice(R, 0, 1):
        lineLis = line.split(',')
        for i in range(len(lineLis)):
            if ('\n' in lineLis[i]) and (lineLis[i] != '\n'):
                headers.append(lineLis[i].replace('\n', ''))
                ans[lineLis[i].replace('\n', '')] = []
            elif (lineLis[i] != '') and (lineLis[i] != '\n'):
                headers.append(lineLis[i])
                ans[lineLis[i]] = []
    R.seek(0)
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(',')
        for i in range(len(headers)):
            ans[headers[i]].append(float(lineLis[i]))
    return ans

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def merge(datUp, datDown, starts):
    ans = {'upstream': {}, 'downstream': {}}
    
    # Slice start is defined by first sample used
    start = starts['sample'][0]
    # Slice end is defined as last samples used plus 2 as slices are non inclusive for the end and we need to average forward
    end = starts['sample'][-1] + 2

    # Fill data keys with relevent keys
    for key in datUp:
        if 'Numb' in key:
            ans['upstream'][key] = []
        elif key == 'Epoch_UTC':
            ans['upstream'][key] = []
    for key in datDown:
        if '_Bin' in key:
            ans['downstream'][key] = datDown[key][start:end - 1]
        elif key == 'Epoch_UTC':
            ans['downstream'][key] = datDown[key][start:end - 1]    
    
    # Assign time arrays for upstream and downstream sensors
    timeUp = np.array(datUp['Epoch_UTC'])
    timeDown = np.array(datDown['Epoch_UTC'])[start:end]

    # Fill list of indicies to match times
    indices = []
    for i in range(len(timeDown)):
        indices.append(find_nearest(timeUp, timeDown[i]))

    # Fill upstream portion of results, subtract privious bin to get size ranges
    privious = [0, 0.5]
    for i in range(len(indices) - 1):
        for key in datUp:
            if 'Numb' in key:
                size = float(key.split('_')[-1])
                start = indices[i]
                end = indices[i + 1]
                temp = np.mean(np.array(datUp[key][start:end]))
                if size > privious[1]:
                    ans['upstream'][key].append(temp - privious[0])
                    privious = [temp, size]
                else:
                    ans['upstream'][key].append(temp)
                    privious = [0, size]
            elif key == 'Epoch_UTC':
                ans['upstream'][key].append(timeUp[indices[i]])

    # Write new answer to return with all the information needed
    temp = ans.copy()
    ans = {}
    for sensor in temp:
        ans[sensor] = {'name': []}
        for key in temp[sensor]:
            ans[sensor][key] = []

    # Write only the results that match the sample idicies
    indices = np.array(starts['sample']) - starts['sample'][0]
    for i in range(len(indices)):
        ans['upstream']['name'].append(starts['name'][i])
        for key in temp['upstream']:
            ans['upstream'][key].append(temp['upstream'][key][indices[i]])
        ans['downstream']['name'].append(starts['name'][i])
        for key in temp['downstream']:
            ans['downstream'][key].append(temp['downstream'][key][indices[i]])

    # Sum across size bins to get data for correction factor
    ans['upstream']['sum'] = []
    ans['downstream']['sum'] = []
    for i in range(len(ans['upstream']['name'])):
        temp = 0
        for key in ans['upstream']:
            if 'Numb' in key:
                temp += ans['upstream'][key][i]
        ans['upstream']['sum'].append(temp)
        temp = 0
        for key in ans['downstream']:
            if '_Bin' in key:
                temp += ans['downstream'][key][i]
        ans['downstream']['sum'].append(temp)
    return ans

def saveDat(allDat, locationSave):
    W = open(locationSave, 'w')
    
    headers = ['name']
    for key in allDat['downstream']:
        if '_Bin' in key:
            headers.append(key)
    W.write((',').join(headers))
    W.write('\n')
    
    for i in range(1, len(allDat['downstream']['name']), 2):
        bypass = []
        material = []
        pen = []
        for j in range(len(headers)):
            bypass.append(str(allDat['downstream'][headers[j]][i - 1]))
            material.append(str(allDat['downstream'][headers[j]][i]))
            if headers[j] == 'name':
                pen.append('N/A')
            else:
                if (allDat['downstream'][headers[j]][i-1] and allDat['downstream'][headers[j]][i+1]) == 0:
                    pen.append('nan')
                else:
                    pen.append(str(allDat['downstream'][headers[j]][i]/(((allDat['downstream'][headers[j]][i - 1])+(allDat['downstream'][headers[j]][i + 1]))/2)))
        W.write((',').join(bypass))
        W.write('\n')
        W.write((',').join(material))
        W.write('\n')
        W.write((',').join(pen))
        W.write('\n')
        xRange, legLis = [], []
        for j in range(1, len(headers)):
            xRange.append(float(headers[j].split('_Bin')[0]))
        plt.plot(xRange, pen[1:])
        legLis.append(allDat['downstream']['name'][i])
    plt.locator_params(axis='y', nbins=6)
    plt.legend(legLis)
    plt.show()
    plt.close()
    W.close()

def objective(x, a, b): 
    return a + b*x

def correctionFact(allDat, locationSave):
    corDat0 = allDat.copy()
    corDat1 = allDat.copy()
    
    # Central correction factor also considered a multiplicitive correction factor
    for i in range(1, len(corDat0['upstream']['name']), 2):
        CF0 = corDat0['upstream']['sum'][i-1]/corDat0['upstream']['sum'][i]
        CF1 = corDat0['upstream']['sum'][i+1]/corDat0['upstream']['sum'][i]
        CF = (CF0+CF1)/2
        for key in corDat0['downstream']:
            if '_Bin' in key:
                temp = corDat0['downstream'][key][i]
                corDat0['downstream'][key][i] *= CF
                #print(corDat0['downstream']['name'][i], temp, corDat0['downstream'][key][i])
    saveDat(corDat0, '%s\%s' %(locationSave, '\Test_048_Out_CF0.csv'))

    # Correction factor with shift in it
    defaults = [0, 1]
    params, cov = curve_fit(objective, corDat1['upstream']['sum'], corDat1['downstream']['sum'], p0 = defaults)
    A, B = params
    for i in range(1, len(corDat1['upstream']['name']), 2):
        CF0 = (B*corDat1['upstream']['sum'][i-1]+A)/(B*corDat1['upstream']['sum'][i]+A)
        CF1 = (B*corDat1['upstream']['sum'][i+1]+A)/(B*corDat1['upstream']['sum'][i]+A)
        CF = (CF0+CF1)/2
        for key in corDat1['downstream']:
            if '_Bin' in key:
                temp = corDat1['downstream'][key][i]
                corDat1['downstream'][key][i] *= CF
                #print(corDat1['downstream']['name'][i], temp, corDat1['downstream'][key][i])
    saveDat(corDat1, '%s\%s' %(locationSave, '\Test_048_Out_CF1.csv'))
    return corDat0, corDat1

def timeSerries(allDat):
    mean_up = np.mean(np.array(allDat['upstream']['sum']))
    mean_down = np.mean(np.array(allDat['downstream']['sum']))
    sum_up = np.array(allDat['upstream']['sum'])/mean_up
    sum_down = np.array(allDat['downstream']['sum'])/mean_down
    time_up = np.array(allDat['upstream']['Epoch_UTC'])-allDat['upstream']['Epoch_UTC'][0]
    time_down = np.array(allDat['downstream']['Epoch_UTC'])-allDat['downstream']['Epoch_UTC'][0]

    bypass_up, bypass_down = [], []
    time_by_up, time_by_down = [], []
    for i in range(0, len(sum_up), 2):
        bypass_up.append(sum_up[i])
        bypass_down.append(sum_down[i])
        time_by_up.append(time_up[i])
        time_by_down.append(time_down[i])

    mat_up, mat_down = [], []
    time_mat_up, time_mat_down = [], []
    for i in range(1, len(sum_up), 2):
        mat_up.append(sum_up[i])
        mat_down.append(sum_down[i])
        time_mat_up.append(time_up[i])
        time_mat_down.append(time_down[i])

    fig, axs = plt.subplots(3)
    fig.suptitle('Particle Count Trends')
    legLis0, legLis1, legLis2 = [], [], []
    UP = True
    for i in range(len(allDat['downstream']['name'])):
        x = allDat['downstream']['Epoch_UTC'][i]-allDat['downstream']['Epoch_UTC'][0]
        y = allDat['downstream']['sum'][i]/mean_down
        name = allDat['downstream']['name'][i]
        if UP:
            axs[0].annotate(name, (x, y), xycoords='data', xytext=(0, 15), textcoords='offset points', size='xx-small')
            axs[0].scatter(x, y)
            UP = False
        else:
            axs[0].annotate(name, (x, y), xycoords='data', xytext=(0, -15), textcoords='offset points', size='xx-small')
            axs[0].scatter(x, y)
            UP = True
    axs[0].plot(time_up, sum_up)
    axs[0].plot(time_down, sum_down)

    UP = True
    for i in range(0, len(allDat['downstream']['name']), 2):
        x = allDat['downstream']['Epoch_UTC'][i]-allDat['downstream']['Epoch_UTC'][0]
        y = allDat['downstream']['sum'][i]/mean_down
        name = allDat['downstream']['name'][i]
        if UP:
            axs[1].annotate(name, (x, y), xycoords='data', xytext=(0, 5), textcoords='offset points', size='xx-small')
            axs[1].scatter(x, y)
            UP = False
        else:
            axs[1].annotate(name, (x, y), xycoords='data', xytext=(0, -5), textcoords='offset points', size='xx-small')
            axs[1].scatter(x, y)
            UP = True
    axs[1].plot(time_by_up, bypass_up)
    axs[1].plot(time_by_down, bypass_down)

    UP = True
    for i in range(1, len(allDat['downstream']['name']), 2):
        x = allDat['downstream']['Epoch_UTC'][i]-allDat['downstream']['Epoch_UTC'][0]
        y = allDat['downstream']['sum'][i]/mean_down
        name = allDat['downstream']['name'][i]
        if UP:
            axs[2].annotate(name, (x, y), xycoords='data', xytext=(0, 5), textcoords='offset points', size='xx-small')
            axs[2].scatter(x, y)
            UP = False
        else:
            axs[2].annotate(name, (x, y), xycoords='data', xytext=(0, -5), textcoords='offset points', size='xx-small')
            axs[2].scatter(x, y)
            UP = True
    axs[2].plot(time_mat_up, mat_up)
    axs[2].plot(time_mat_down, mat_down)
    plt.show()
    plt.close()
    

# Get user input for file locations
srcPath = '.\Data\CSV'
locationUp = 'SEN_2021-08-06_12-03-37-SPS3x_Test_048.csv'   #input('Enter File Name for Upstream Sensor: ')
locationDown = 'TSI_2021_08_06-12_06_18-TEST_048.csv'       #input('Enter File Name for Downstream Sensor: ')
locationStarts = 'Starts_Press_TEST_048.csv'                #input('Enter File Name For Starts File: ')
locationSave = '.\Data\CSV\Output'
locationUp = '%s\%s' %(srcPath, locationUp)
locationDown = '%s\%s' %(srcPath, locationDown)
locationStarts = '%s\%s' %(srcPath, locationStarts)
starts = readStarts(locationStarts)
datUp = readDat(locationUp)
datDown = readDat(locationDown)
allDat = merge(datUp, datDown, starts)
timeSerries(allDat)
for i, key in enumerate(allDat['upstream']['name']):
    if (key == 'bypass'):
        plt.scatter(allDat['upstream']['sum'][i], allDat['downstream']['sum'][i], color='b')
    else:
        plt.scatter(allDat['upstream']['sum'][i], allDat['downstream']['sum'][i], color='r')
plt.show()
plt.close()
saveDat(allDat, '%s\%s' %(locationSave, '\Test_048_Out.csv'))
corDat0, corDat1 = correctionFact(allDat, locationSave)