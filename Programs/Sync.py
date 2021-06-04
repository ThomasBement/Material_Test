# ---------------------------------------- #
# Sync [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-12
# ---------------------------------------- #

"""
IMPORTS
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
READING DATA
"""
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

"""
COMBINING BOTH DATA SOURCES INTO ONE OBJECT
"""
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
            stop = datLis[i][1][1]
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
    return ans

"""
FORMATTING AND WRITING DATA
"""

# Reitteration of readDict function for the time .CSV format
def readTime(location):
    R = open(location, 'r')
    ans = {}
    R.seek(0)
    Headers = R.readline().split(',')
    Headers[-1] = Headers[-1].replace('\n', '')
    R.seek(0)
    for i in range(len(Headers)):
        ans[Headers[i]] = []
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(',')
        for i in range(len(Headers)):
            ans[Headers[i]].append(float(lineLis[i]))
    for key in ans:
        ans[key] = np.array(ans[key])
    return ans

# Find nearest value in an array and return its index
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# saveDat is very bloated and yucky but it works, try to fix it when you have time
def saveDat(dat, times, location, sensorNames, Numb, writeLis = ['Bin']):
    # Choose the right mesurement based on user input
    if Numb:
        writeLis.append('Numb')
    else:
        writeLis.append('Mass')

    headersUp, headersDown = {}, {}
    # Write the headers and top info block
    for i in range(len(times['Start_Time'])):
        for sensor in dat:
            time = datetime.utcfromtimestamp(int(times['Start_Time'][i])).strftime('%Y-%m-%d_%H_%M_%S')
            runTime = int(times['Stop_Time'][i])-int(times['Start_Time'][i])
            W = open('%s\%s_%s.csv' %(location, sensor, time), 'w')
            W.write('Test Number: %s\n' %i)
            W.write('Sensor: %s\n' %sensorNames[sensor])
            W.write('Start Time: %s\n' %times['Start_Time'][i])
            W.write('Stop Time: %s\n' %times['Stop_Time'][i])
            W.write('Run Time: %s\n' %str(runTime))
            if ('Up' in sensor):
                headersUp[sensor] = ['Samples', 'Epoch_UTC']
                for mesurment in dat[sensor]:
                        for string in writeLis:
                            if (string in mesurment):
                                headersUp[sensor].append(mesurment)
                W.write(','.join(headersUp[sensor]))
            else:
                headersDown[sensor] = ['Samples', 'Epoch_UTC']
                for mesurment in dat[sensor]:
                        for string in writeLis:
                            if (string in mesurment):
                                headersDown[sensor].append(mesurment)
                W.write(','.join(headersDown[sensor]))
            W.close()
    # Assign keys from upstream to downstream
    binsUp = []
    keysUp = []
    keysDown = {}
    
    # For the upstream sensor we write the numerical bin sizes to an array
    for key in headersUp['UpStream']:
        header = key.split(writeLis[1]+'Conc_')[-1]
        if not ((header == 'Samples') or (header == 'Epoch_UTC')):
            binsUp.append(float(header))
            keysUp.append(key)
    binsUp = np.array(binsUp)
    # For the downstream sensor we add the indicies of the closest size bins from the upstream array
    for key in headersDown['DownStream']:
        header = key.split('_'+writeLis[0])[0]
        if not ((header == 'Samples') or (header == 'Epoch_UTC')):
            keysDown[key] = find_nearest(binsUp, float(header))
    # Peroform the writting for the correection factors        
    for i in range(1, len(times['Start_Time']), 2):
        # Get two times for opening files (one is the reference frame from which the calibration factor is found)
        time0 = datetime.utcfromtimestamp(int(times['Start_Time'][i-1])).strftime('%Y-%m-%d_%H_%M_%S')
        time1 = datetime.utcfromtimestamp(int(times['Start_Time'][i])).strftime('%Y-%m-%d_%H_%M_%S')
        print(times['Start_Time'][i-1], times['Stop_Time'][i-1])
        print(times['Start_Time'][i], times['Stop_Time'][i])
        print('\n')
        start0 = np.where(dat[sensor]['Epoch_UTC'] >= times['Start_Time'][i-1])[0][0]
        stop0 = np.where(dat[sensor]['Epoch_UTC'] >= times['Stop_Time'][i-1])[0][0]
        start1 = np.where(dat[sensor]['Epoch_UTC'] >= times['Start_Time'][i])[0][0]
        stop1 = np.where(dat[sensor]['Epoch_UTC'] >= times['Stop_Time'][i])[0][0]
        # Open all 4 output files
        upA0 = open('%s\%s_%s.csv' %(location, 'UpStream', time0), 'a')
        upA1 = open('%s\%s_%s.csv' %(location, 'UpStream', time1), 'a')
        downA0 = open('%s\%s_%s.csv' %(location, 'DownStream', time0), 'a')
        downA1 = open('%s\%s_%s.csv' %(location, 'DownStream', time1), 'a')
        upA0.write('\n')
        upA1.write('\n')
        downA0.write('\n')
        downA1.write('\n')
        # Assign lists for correction factors for upstream sensor
        correctionFact0, correctionFact1 = [], []
        for key in headersUp['UpStream']:
            if ((key == 'Samples') or (key == 'Epoch_UTC')):
                correctionFact0.append('1')
                correctionFact1.append('1')
            else:
                correctionFact0.append('1')
                temp = np.mean(dat['UpStream'][key][start0:stop0])/np.mean(dat['UpStream'][key][start1:stop1])
                correctionFact1.append(str(temp))
        upA0.write(','.join(correctionFact0))
        upA1.write(','.join(correctionFact1))
        upA0.close()
        upA1.close()
        # Assign lists for correction factors for downstream sensor
        correctionFact0, correctionFact1 = [], []
        for key in headersDown['DownStream']:
            if ((key == 'Samples') or (key == 'Epoch_UTC')):
                correctionFact0.append('1')
                correctionFact1.append('1')
            else:
                # temp can be cleaned up to reuse the privious correction factors with the index keysDown[key]
                correctionFact0.append('1')
                temp = np.mean(dat['UpStream'][keysUp[keysDown[key]]][start0:stop0])/np.mean(dat['UpStream'][keysUp[keysDown[key]]][start1:stop1])
                correctionFact1.append(str(temp))
        downA0.write(','.join(correctionFact0))
        downA1.write(','.join(correctionFact1))
        downA0.close()
        downA1.close() 
    # Loop through sensors and write the rest of the data
    for i in range(len(times['Start_Time'])):
        for sensor in dat:
            time = datetime.utcfromtimestamp(int(times['Start_Time'][i])).strftime('%Y-%m-%d_%H_%M_%S')
            start = np.where(dat[sensor]['Epoch_UTC'] >= times['Start_Time'][i])[0][0]
            stop = np.where(dat[sensor]['Epoch_UTC'] >= times['Stop_Time'][i])[0][0]
            A = open('%s\%s_%s.csv' %(location, sensor, time), 'a')
            A.write('\n')           
            for j in range(start, stop):
                line = [str(j)]
                if ('Up' in sensor):
                    for key in headersUp[sensor]:                        
                        if (key != 'Samples'):
                            line.append(str(dat[sensor][key][j]))
                else:
                    for key in headersDown[sensor]:
                        if (key != 'Samples'):
                            line.append(str(dat[sensor][key][j]))
                A.write(','.join(line))
                A.write('\n')
            A.close()

            
"""
STATISTICS FUNCTIONS
"""
def coeffDeterm(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def pearsonr(x, y):
    # Assume len(x) == len(y)
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(xi*xi for xi in x)
    sum_y_sq = sum(yi*yi for yi in y)
    psum = sum(xi*yi for xi, yi in zip(x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

"""
PLOTTING FUNCTIONS
"""
def compareArr(arrX1, arrX2, arrY1, arrY2, boxSz = 5):
    if (len(arrY1) != len(arrY2)):
        print('Array sized do not match')
        quit()
    else:
        arrComp = []
        arrX = []
        for i in range(boxSz, len(arrY1)):
            avgY = [0, 0]
            avgX = [0, 0]
            for j in range(boxSz):
                avgY[0] += arrY1[i-j]/boxSz
                avgY[1] += arrY2[i-j]/boxSz
                avgX[0] += arrX1[i-j]/boxSz
                avgX[1] += arrX2[i-j]/boxSz
            arrX.append((avgX[0]+avgX[1])/2)
            arrComp.append(abs(avgY[0]-avgY[1]))
        return arrX, arrComp

def timeSerries(allDat, inKey, saveAs):
    legendLis = ['SEN_1.0 um', 'TSI_1.0 um', 'Subtracted Means']
    X1 = allDat['UpStream']['Epoch_UTC']
    X2 = allDat['DownStream']['Epoch_UTC']
    Y1 = allDat['UpStream']['NumbConc_1.0']
    Y2 = allDat['DownStream']['1.117_Bin']
    # For Z-statistic
    #X3, Y3 = compareArr(X1, X2, Y1, Y2, 24)
    #for nameKeys in allDat:
    #    for keys in allDat[nameKeys]:
    #        for elem in inKey: 
    #            if (elem in keys):
    #                plt.plot(allDat[nameKeys]['Epoch_UTC'], allDat[nameKeys][keys])
    #                legendLis.append(nameKeys + ', ' + keys)   
    #plt.plot(X3, Y3)
    plt.title('Z-Statistic:')
    plt.plot(X1, Y1)
    plt.plot(X2, Y2)
    plt.legend(legendLis, bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.savefig(saveAs, format='png', bbox_inches='tight')
    plt.show()
    plt.close()
    return 0

"""
MAIN
"""
def main():
    # Get user inputs for each source file, in future a single folder could be used
    print('Enter Parameters for Upstream Sensor:')
    locationUp = input('Enter File Name: ')
    locationUp = '.\Data\CSV\\' + locationUp
    print('Enter Parameters for Downstream Sensor:')
    locationDown = input('Enter File Name: ')
    # User needs to specify what they set the TSI sensor to
    mesurmentTyp = input('Enter Mesuremnt Type [Mass/Number]: ')
    Numb = True
    if (mesurmentTyp == 'Mass'):
        Numb = False
    locationDown = '.\Data\CSV\\' + locationDown
    print('Enter Parameters for Time Log:')
    locationTime = input('Enter File Name: ')
    locationTime = '.\Data\CSV\\' + locationTime
    print('Save Location:')
    loacationSave = input('Enter Save Folder Path: ')
    # Define sensor names for future use
    sensorNames = {'UpStream': 'Unknown', 'DownStream': 'TSI_3330_Opticle_Particle_Sizer'}
    if ('SEN_' in locationUp):
        sensorNames['UpStream'] = 'SPS30'
    elif ('PMS_' in locationUp):
        sensorNames['UpStream'] = 'PMS-5003'
    # Define location to save figure to
    print('Ploting Information:')
    saveAs = input('Enter Save Figure Path: ')
    if (saveAs == ''):
        saveAs = r'.\Data\Figures\Untitled_001.png'
    # Read data and merge
    datUp = readDict(locationUp)
    datDown = readDict(locationDown)
    allDat = mergeDat(datUp, datDown, ['UpStream', 'DownStream'])
    timeSplits = readTime(locationTime)
    saveDat(allDat, timeSplits, loacationSave, sensorNames, Numb)
    timeSerries(allDat, ['1.0_Bin', 'NumbConc_1.0'], saveAs)
    return 0

# Call main once, for performing analysis on large groups of data 
# main can be edited to be called on every file in a folder
main()