# ---------------------------------------- #
# CorrectionFact002 [Python File]
# Written By: Thomas Bement
# Created On: 2021-06-18
# ---------------------------------------- #

import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

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

# Gets range of usable data where mesurements are close together
def getRange(dat1, dat2):
    minimum = max(dat1['Epoch_UTC'][0], dat2['Epoch_UTC'][0])
    maximum = min(dat1['Epoch_UTC'][-1], dat2['Epoch_UTC'][-1])
    dat1_rng = [np.argwhere(dat1['Epoch_UTC'] >= minimum)[0][0], np.argwhere(dat1['Epoch_UTC'] >= maximum)[0][0]]
    dat2_rng = [np.argwhere(dat2['Epoch_UTC'] >= minimum)[0][0], np.argwhere(dat2['Epoch_UTC'] >= maximum)[0][0]]
    return dat1_rng, dat2_rng

# Average y2 to have time base of x1
def merge(x1, x2, y2):
    i2 = 0
    totals = np.zeros_like(x1)
    counts = np.zeros_like(x1, dtype=int)
    for i1 in range(x1.shape[0] - 1):
        while 2 * x2[i2] < x1[i1+1] + x1[i1]:
            totals[i1] += y2[i2]
            counts[i1] += 1
            i2 += 1
    while i2 < x2.shape[0]:
        totals[-1] += y2[i2]
        counts[-1] += 1
        i2 += 1
    return totals / counts

# Align time data and merge data points so sensor data matches number of samples
def mergeDat(dat1, dat2, keys):
    dat1_rng, dat2_rng = getRange(dat1, dat2)
    datLis = [(dat1, dat1_rng), (dat2, dat2_rng)]
    # Trim data which is not mesured in the same time interval
    for i in range(len(datLis)):
        for key in datLis[i][0]:
            start = datLis[i][1][0]
            # Add 1 to prevent off by one error from slice
            stop = datLis[i][1][1] + 1
            datLis[i][0][key] = datLis[i][0][key][start:stop]
    # Group data points from the same sample time together
    temp = {}
    ans = {}
    ans[keys[0]], ans[keys[1]] = {}, {}
    temp[keys[0]], temp[keys[1]] = {}, {}
    i = 0
    for nameKey in keys:
        for key in datLis[i][0]: 
            temp[nameKey][key] = []  
            for j in range(len(datLis[i][0][key])):
                if (key == 'Epoch_UTC'):
                    temp[nameKey][key].append(datLis[i][0][key][j])
                else:    
                    temp[nameKey][key].append(datLis[i][0][key][j])
        i += 1
    
    for nameKey in temp:
        for key in temp[nameKey]:
            temp[nameKey][key] = np.array(temp[nameKey][key])
    # Pick data to keep (least number of samples)
    keep = keys[0]
    scale = keys[1]
    if (len(temp[keep]['Epoch_UTC']) > len(temp[scale]['Epoch_UTC'])):
        keep = keys[1]
        scale = keys[0]
    # Write data to be kept to ans
    for key in temp[keep]:
        ans[keep][key] = temp[keep][key]
    # Group and write other data
    for key in temp[scale]:
        ans[scale][key] = merge(temp[keep]['Epoch_UTC'], temp[scale]['Epoch_UTC'], temp[scale][key])
    return ans, max(dat1_rng[0], dat2_rng[0])

# Reads starts .csv to get index, name and pressure of each test
def readStarts(locationStarts):
    R = open(locationStarts, 'r')
    ans = {}
    R.seek(0)
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(',')
        ans[str(int(lineLis[2])-1)] = [lineLis[0], lineLis[1]]
    return ans

# Returns index of value in array nearest to a given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def objective(x, a, b): 
    return a + b*x

def saveDat(allDat, starts, mesurmentTyp, sensorTyp, locationDown, idx1):
    temp = {}
    for sensor in allDat:
        temp[sensor] = []
        for key in allDat[sensor]:
            if (sensor == 'UpStream'):
                if (mesurmentTyp in key):
                    temp[sensor].append([])
                    for i in range(len(allDat[sensor][key])):
                        temp[sensor][-1].append(allDat[sensor][key][i])
                elif (key == 'Epoch_UTC'):
                    for i in range(len(allDat[sensor][key])):
                        allDat[sensor][key][i] = allDat[sensor][key][i] - 10

            else:
                if ('Bin' in key):
                    temp[sensor].append([])
                    for i in range(len(allDat[sensor][key])):
                        temp[sensor][-1].append(allDat[sensor][key][i])

    totals = {}
    for sensor in temp:
        totals[sensor] = []
        for i in range(len(temp[sensor][0])):
            total = 0
            for j in range(len(temp[sensor])):
                total += temp[sensor][j][i]
            totals[sensor].append(total)

    defaults = [0, 1]
    params, cov = curve_fit(objective, totals['UpStream'][1:], totals['DownStream'][:-1], p0 = defaults)
    A, B = params
    xRange = np.linspace(min(totals['UpStream'][1:]), max(totals['UpStream'][1:]), 256)
    yRange = np.zeros(256)
    for i in range(len(xRange)):
        yRange[i] = objective(xRange[i], *params)

    temp = {}
    for sensor in allDat:
        temp[sensor] = []
        for idx in starts:
            print(starts[idx][0], int(idx)-idx1, len(totals[sensor]))
            temp[sensor].append([starts[idx][0], totals[sensor][(int(idx)-idx1-1)], int(idx)])

    for i in range(len(temp['UpStream'])):
        plt.scatter(temp['UpStream'][i][1], temp['DownStream'][i][1])
    plt.plot(xRange, yRange)
    plt.show()
    plt.close()

    for i in range(2, len(temp['UpStream']), 2):
        CF0 = (B*temp['UpStream'][i-1][1]+A)/(B*temp['UpStream'][i-2][1]+A)
        CF1 = (B*temp['UpStream'][i-1][1]+A)/(B*temp['UpStream'][i][1]+A)
        CF = (CF0+CF1)/2
        temp['DownStream'][i-1].append(CF)

    test_num = ('_'.join([locationDown.split('_')[-2].split('-')[-1], locationDown.split('_')[-1].split('.')[0]]))
    reg = open('.\Data\CSV\Output\%s_Output_Reg.csv' %test_num, 'w')
    cor = open('.\Data\CSV\Output\%s_Output_Cor.csv' %test_num, 'w')
    # Fill headers 
    headers = ['Name', 'Value', 'Sample']
    bins = []
    for key in allDat['DownStream']:
        if '_Bin' in key:
            headers.append(key)
            bins.append(key)
    # Write headers
    reg.write((',').join(headers))
    reg.write('\n')
    cor.write((',').join(headers))
    cor.write('\n')
    # Write regular output
    for i in range(2, len(temp['DownStream']), 2):
        line0 = ['bypass', 'Number concentration (#/cc)', str(temp['DownStream'][i-2][2])]
        line1 = [str(temp['DownStream'][i-1][0]), 'Number concentration (#/cc)', str(temp['DownStream'][i-1][2])]
        for key in bins:
            line0.append(str(allDat['DownStream'][key][i-2]))
            line1.append(str(allDat['DownStream'][key][i-1]))
        reg.write((',').join(line0))
        reg.write('\n')
        reg.write((',').join(line1))
        reg.write('\n')
    # Write correction factor output
    for i in range(2, len(temp['DownStream']), 2):
        line0 = ['bypass', 'Number concentration (#/cc)', str(temp['DownStream'][i-2][2])]
        line1 = [str(temp['DownStream'][i-1][0]), 'Number concentration (#/cc)', str(temp['DownStream'][i-1][2])]
        for key in bins:
            line0.append(str(allDat['DownStream'][key][i-2]))
            line1.append(str(allDat['DownStream'][key][i-1]*temp['DownStream'][i-1][3]))
        cor.write((',').join(line0))
        cor.write('\n')
        cor.write((',').join(line1))
        cor.write('\n')
    # Close the write files
    reg.close()
    cor.close()
    #CF = (A x_filter + B)/(A x_bypass + B)
    return totals
               
def main():
    # Get user inputs for each source file, in future a single folder could be used
    print('Enter Parameters for Upstream Sensor:')
    locationUp = input('Enter File Name: ')
    sensorTyp = locationUp.split('_')[0]
    locationUp = '.\Data\CSV\\' + locationUp
    print('Enter Parameters for Downstream Sensor:')
    locationDown = input('Enter File Name: ')
    locationDown = '.\Data\CSV\\' + locationDown
    # User needs to specify what they set the TSI sensor to
    mesurmentTyp = 'Numb'#input('Enter Mesuremnt Type [Mass/Numb]: ')
    # Get file name for the start information
    locationStarts = input('Enter File Name For Starts File: ')
    locationStarts = '.\Data\CSV\\' + locationStarts
    datUp = readDict(locationUp)
    datDown = readDict(locationDown)
    allDat, idx1 = mergeDat(datUp, datDown, ['UpStream', 'DownStream'])
    starts = readStarts(locationStarts)
    saveDat(allDat, starts, mesurmentTyp, sensorTyp, locationDown, idx1)
    
main()