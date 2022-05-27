# ---------------------------------------- #
# CorrectionFact [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-17
# ---------------------------------------- #

import itertools
import numpy as np
import matplotlib.pyplot as plt
import math

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
    return ans

# Reads starts .csv to get index, name and pressure of each test
def readStarts(locationStarts):
    R = open(locationStarts, 'r')
    ans = {}
    R.seek(0)
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(',')
        ans[str(int(lineLis[2]) - 1)] = [lineLis[0], lineLis[1]]
    return ans

# Returns index of value in array nearest to a given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def saveDat(allDat, starts, loacationSave, mesurmentTyp, sensorTyp):
    temp = {}
    for sensor in allDat:
        temp[sensor] = []
        for idx in starts:
            temp[sensor].append((starts[idx][0],[],[], float(starts[idx][1])))
            for key in allDat[sensor]:
                # More conditions for PMS sensor may be needed, switching to only using numb in code would be cleaner
                if sensorTyp == 'SEN':
                    if (mesurmentTyp in key) or (sensor == 'DownStream'):
                        temp[sensor][-1][1].append(key)
                        temp[sensor][-1][2].append(allDat[sensor][key][int(idx)-1])
                if sensorTyp == 'PMS':
                    if ('Bin' in key) or (sensor == 'DownStream'):
                        temp[sensor][-1][1].append(key)
                        temp[sensor][-1][2].append(allDat[sensor][key][int(idx)-1])
                # Proof that the seccond minus one is needed to avound off by one error
                #if (key == 'Epoch_UTC'):
                #    print(allDat[sensor][key][int(idx)-1])
    # Generate bin overlap keys
    UpStream_Keys = ([], [], [])
    if sensorTyp == 'SEN':
        for i in range(len(temp['UpStream'][0][1])):
            if (mesurmentTyp in temp['UpStream'][0][1][i]):
                UpStream_Keys[0].append(temp['UpStream'][0][1][i])
                UpStream_Keys[1].append(float(temp['UpStream'][0][1][i].split('_')[-1]))
                UpStream_Keys[2].append(i)
    if sensorTyp == 'PMS':
        for i in range(len(temp['UpStream'][0][1])):
            if ('Bin' in temp['UpStream'][0][1][i]):
                UpStream_Keys[0].append(temp['UpStream'][0][1][i])
                UpStream_Keys[1].append(float(temp['UpStream'][0][1][i].split('_')[0]))
                UpStream_Keys[2].append(i)
    # Use bin sizes from upstream to find shred bins for downstream and write all info needed for relating the two
    DownStream_Keys = ([], [], [], [], [])
    for i in range(len(temp['DownStream'][0][1])):
            if ('Bin' in temp['DownStream'][0][1][i]):
                DownStream_Keys[0].append(temp['DownStream'][0][1][i])
                DownStream_Keys[1].append(float(temp['DownStream'][0][1][i].split('_')[0]))
                DownStream_Keys[2].append(UpStream_Keys[0][find_nearest(UpStream_Keys[1], float(temp['DownStream'][0][1][i].split('_')[0]))])
                DownStream_Keys[3].append(i)
                DownStream_Keys[4].append(UpStream_Keys[2][find_nearest(UpStream_Keys[1], float(temp['DownStream'][0][1][i].split('_')[0]))])
    # Write results to dictonary for final formatting
    results = {}
    for i in range(2, len(temp['DownStream']), 2):
        # Assign lists to store information in for results (Labels, CF, CountM, CountNM, press)
        results[temp['DownStream'][i-1][0]] = [DownStream_Keys[0], [], [], [], temp['DownStream'][i-1][3]]
        CF, countsM, countsNM = [], [], []
        for j in range(len(DownStream_Keys[0])):
            CF.append(((temp['UpStream'][i-2][2][DownStream_Keys[4][j]]/temp['UpStream'][i-1][2][DownStream_Keys[4][j]]) + (temp['UpStream'][i][2][DownStream_Keys[4][j]]/temp['UpStream'][i-1][2][DownStream_Keys[4][j]]))/2)
            countsM.append(temp['DownStream'][i-1][2][DownStream_Keys[3][j]])
            countsNM.append((temp['DownStream'][i-2][2][DownStream_Keys[3][j]] + temp['DownStream'][i][2][DownStream_Keys[3][j]])/2)
            if  (DownStream_Keys[4][j] == 0):
                # Particles in the 0.3-0.5 range
                CF.append(((temp['UpStream'][i-2][2][DownStream_Keys[4][j]]/temp['UpStream'][i-1][2][DownStream_Keys[4][j]]) + (temp['UpStream'][i][2][DownStream_Keys[4][j]]/temp['UpStream'][i-1][2][DownStream_Keys[4][j]]))/2)
            else:
                # Particles need privious bin subtracted to remove the 0.3 to the lower bound particles 
                top_0 = temp['UpStream'][i-2][2][DownStream_Keys[4][j]] - temp['UpStream'][i-2][2][DownStream_Keys[4][j]-1]
                bot_0 = temp['UpStream'][i-1][2][DownStream_Keys[4][j]] - temp['UpStream'][i-1][2][DownStream_Keys[4][j]-1]
                top_1 = temp['UpStream'][i][2][DownStream_Keys[4][j]] - temp['UpStream'][i][2][DownStream_Keys[4][j]-1]
                bot_1 = temp['UpStream'][i-1][2][DownStream_Keys[4][j]] - temp['UpStream'][i-1][2][DownStream_Keys[4][j]-1]
                CF.append(((top_0/bot_0) + (top_1/bot_1))/2)
                
        results[temp['DownStream'][i-1][0]][1] = CF
        results[temp['DownStream'][i-1][0]][2] = countsM
        results[temp['DownStream'][i-1][0]][3] = countsNM
    # Open output file and write results to it
    W = open(loacationSave, 'w')
    for name in results:
        W.write(name + '\n')
        headers, CF, pen, eff, qual = [], [], [], [], []
        for i in range(len(results[name][0])):
            headers.append(str(results[name][0][i]))
            CF.append(str(results[name][1][i]))
            pen.append(str(results[name][1][i]*results[name][2][i]/results[name][3][i]))
            eff.append(str(1-(results[name][1][i]*results[name][2][i]/results[name][3][i])))
            if (results[name][1][i]*results[name][2][i]/results[name][3][i] <= 0):
                qual.append('nan')
            else:
                qual.append(str((-1*math.log(results[name][1][i]*results[name][2][i]/results[name][3][i]))/results[name][4]))
        W.write((',').join(headers) + '\n')
        W.write((',').join(CF) + '\n')
        W.write((',').join(pen) + '\n')
        W.write((',').join(eff) + '\n')
        #W.write((',').join(qual) + '\n')
    W.close()
    return results

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def compareBox(dat, layers):
    ticks = []
    stats = {}
    for key in dat[layers]:
        stats[key] = {'Mean': [], 'Std': []}
        ticks.append(key.split('_')[0])
        for i in range(len(dat[layers][key])):
            dat[layers][key][i] =  np.array(dat[layers][key][i])
    ticks_a, ticks_b, ticks_float = [float(ticks[0]) + (float(ticks[1])-float(ticks[0]))/4], [float(ticks[0]) - (float(ticks[1])-float(ticks[0]))/4], [float(ticks[0])]
    for i in range(1, len(ticks)):
        offset = (float(ticks[i])-float(ticks[i-1]))/4
        ticks_a.append(float(ticks[i]) + offset)
        ticks_b.append(float(ticks[i]) - offset)
        ticks_float.append(float(ticks[i]))
        ticks[i] = '%.2f' %float(ticks[i])
    # Scale data to a z-statistic
    data_a = []
    data_b = []
    for key in dat[layers]:
        data_a.append([])
        data_b.append([])
        for i in range(len(dat[layers][key][0])):
            data_a[-1].append(1-(dat[layers][key][1][i]/dat[layers][key][2][i]))
            data_b[-1].append(1-(dat[layers][key][0][i]*dat[layers][key][1][i]/dat[layers][key][2][i]))    
    stats_a = [[], [], []]
    stats_b = [[], [], []]
    for i in range(len(data_a)):
        data_a[i] = np.array(data_a[i])
        data_b[i] = np.array(data_b[i])
        stats_a[0].append(ticks[i])
        stats_b[0].append(ticks[i])
        stats_a[1].append(np.mean(data_a[i]))
        stats_b[1].append(np.mean(data_b[i]))
        stats_a[2].append(np.std(data_a[i]))
        stats_b[2].append(np.std(data_b[i]))
    
    plt.figure()
    plt.title('Visualization of Mean and STD of Efficencies for %s' %layers, fontsize=25)
    plt.xlabel('Particle Size [um]', fontsize=20)
    plt.ylabel('Filtration Efficency [%]', fontsize=20)
    plt.ylim(-0.10, 1.10)
    plt.plot([], c='#D7191C', label='Regular')
    plt.plot([], c='#2C7BB6', label='Corrected')
    plt.legend()

    error_a, error_b = [], []
    means_a, means_b = [], []
    for i in range(len(stats_a[0])):
        error_a.append(stats_a[2][i])
        error_b.append(stats_b[2][i])
        means_a.append(stats_a[1][i])
        means_b.append(stats_b[1][i])

    plt.errorbar(ticks_a, means_a, yerr = error_a, xerr = None, linestyle='', marker ='s', markersize = 3, capsize=2, color = '#D7191C')
    plt.errorbar(ticks_b, means_b, yerr = error_b, xerr = None, linestyle='', marker ='s', markersize = 3, capsize=2, color = '#2C7BB6')
    
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xticks(ticks_float[:14], ticks[:14], rotation=90, ) 

    plt.show()
    plt.close()

    plt.figure()
    plt.title('Box Plot for %s:' %layers)
    plt.xlabel('Particle Size [um]')
    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.4)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.4)
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Regular')
    plt.plot([], c='#2C7BB6', label='Corrected')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(0, 8)
    plt.tight_layout()
    #plt.savefig('BoxCompare_%s.png' %layers)
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
    #plt.show()
    plt.close()

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
    Numb = True
    if (mesurmentTyp == 'Mass'):
        Numb = False
    print('Save Location:')
    loacationSave = input('Enter Save Folder Path: ')
    datUp = readDict(locationUp)
    datDown = readDict(locationDown)
    allDat = mergeDat(datUp, datDown, ['UpStream', 'DownStream'])
    starts = readStarts(locationStarts)
    results = saveDat(allDat, starts, loacationSave, mesurmentTyp, sensorTyp)
    plt_results = {'SB1-1x': {}, 'SB1-3x': {}, 'SB1-5x': {}}
    bins = results['SB1-1x'][0]
    for key in results:
        if 'SB1-1x' in key:
            for i in range(len(bins)):
                plt_results['SB1-1x'][bins[i]] = [[], [], []]
        elif 'SB1-3x' in key:
            for i in range(len(bins)):
                plt_results['SB1-3x'][bins[i]] = [[], [], []]
        elif 'SB1-5x' in key:
            for i in range(len(bins)):
                plt_results['SB1-5x'][bins[i]] = [[], [], []]
    for key in results:
        if 'SB1-1x' in key:
            for i in range(len(bins)):
                plt_results['SB1-1x'][bins[i]][0].append(results[key][1][i])
                plt_results['SB1-1x'][bins[i]][1].append(results[key][2][i])
                plt_results['SB1-1x'][bins[i]][2].append(results[key][3][i])
        if 'SB1-3x' in key:
            for i in range(len(bins)):
                plt_results['SB1-3x'][bins[i]][0].append(results[key][1][i])
                plt_results['SB1-3x'][bins[i]][1].append(results[key][2][i])
                plt_results['SB1-3x'][bins[i]][2].append(results[key][3][i])
        if 'SB1-5x' in key:
            for i in range(len(bins)):
                plt_results['SB1-5x'][bins[i]][0].append(results[key][1][i])
                plt_results['SB1-5x'][bins[i]][1].append(results[key][2][i])
                plt_results['SB1-5x'][bins[i]][2].append(results[key][3][i])
    
    compareBox(plt_results, 'SB1-1x')
    
main()