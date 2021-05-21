# ---------------------------------------- #
# ReadFMT [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-17
# ---------------------------------------- #
"""
IMPORTS
"""
import time
import datetime
import itertools

"""
READING IN SPECIFIC FORMATS
"""
# Converts the Sensitron data to a standard format CSV
def readSensitron(path, filName, headers = 9, delim = '	'):
    # Headers: row 9
    print('Reading: %s\%s' %(path, filName))
    name = filName.split('.')[0]
    W = open('.\Data\CSV\SEN_%s.csv' %name, 'w')
    with open('%s\%s' %(path, filName), 'r') as R:
        firstLine = True
        for line in itertools.islice(R, headers, None):
            if firstLine:
                lineLis = line.split(delim)
                for i in range(len(lineLis)):
                    if ('SPS3x' in lineLis[i]):
                        # Remove long unnecicary parts of headers
                        newHeader = lineLis[i].split('SPS3x')
                        lineLis[i] = newHeader[0]
                        # Trim remaining _
                        newHeader = lineLis[i].split('_')
                        lineLis[i] = '_'.join(newHeader[:-1])
                    if ('Conc' in lineLis[i]):
                        lineLis[i] = lineLis[i].replace('p', '.')
                newLine = ','.join(lineLis)
                W.write(newLine)
                firstLine = False
                W.write('\n')
            else:
                newLine = line.replace(delim, ',')
                W.write(newLine)
    W.close()
    return '.\Data\CSV\SEN_%s.csv' %name

# Read files in TSI format
def readTSI(path, filName, bins = [11, 28], headers = 37, start = 6, date = 7, delim = ','):
    #timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    #print(timestamp)
    W = open('%s\TSI_%s' %(path, filName), 'w')
    print('Reading: %s\%s' %(path, filName))
    with open('%s\%s' %(path, filName)) as R:
        Bins = []
        firstLine = True
        timeIdx = 0
        R.seek(0)
        # Get time information
        for line in itertools.islice(R, start, start + 1):
            timeLis = (line.split(',')[-1].split(':'))
            for i in range(len(timeLis)):
                timeLis[i] = int(timeLis[i])
            sec_aft_mn = (3600*timeLis[0])+(60*timeLis[1])+(timeLis[2])
        R.seek(0)
        # Get date information
        for line in itertools.islice(R, date, date + 1):
            temp = line.split(',')[-1].replace('\n', '')
            dateUNIX = time.mktime(datetime.datetime.strptime(temp, "%Y/%m/%d").timetuple())
        startUNIX = dateUNIX + sec_aft_mn
        R.seek(0)
        # Get bin sizes
        for line in itertools.islice(R, bins[0], bins[1]):
            if (line.split(',')[-1] == '\n'):
                temp = 0
            else:
                temp = float(line.split(',')[-1])
            Bins.append(temp)
        R.seek(0)
        for line in itertools.islice(R, headers, None):
            if firstLine:
                lineLis = line.split(delim)
                binIdx = 0
                for i in range(len(lineLis)):
                    # Add bin size into header
                    if ('Bin' in lineLis[i]):
                        lineLis[i] = '%s_Bin' %Bins[binIdx]
                        binIdx += 1
                    # Remove Spaces from header
                    if (' ' in lineLis[i]):
                        lineLis[i] = lineLis[i].replace(' ', '_')
                    if (('Elapsed' and 'Time') in lineLis[i]):
                        lineLis[i] = 'Epoch_UTC'
                        timeIdx = i 
                newLine = ','.join(lineLis)
                W.write(newLine)
                firstLine = False
            else:
                lineLis = line.split(delim)
                for i in range(len(lineLis)):
                    try:
                        lineLis[i] = float(lineLis[i])
                    except:
                        lineLis[i] = 0.0
                lineLis[timeIdx] += startUNIX
                for i in range(len(lineLis)):
                    lineLis[i] = str(lineLis[i])
                newLine = ','.join(lineLis)
                W.write(newLine)
                W.write('\n')        
    W.close()
    return '%s\TSI_%s' %(path, filName)

print('Enter Parameters for Upstream Sensor:')
pathUp = input('Enter Path Location: ')
filNameUp = input('Enter File Name: ')
print('Enter Parameters for Downstream Sensor:')
pathDown = input('Enter Path Location: ')
filNameDown = input('Enter File Name: ')

# Write upstream Data
readSensitron(pathUp, filNameUp)
# Write downstream Data
readTSI(pathDown, filNameDown)