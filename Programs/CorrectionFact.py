# ---------------------------------------- #
# CorrectionFact [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-17
# ---------------------------------------- #

import itertools
import numpy as np
import os
import glob
import time
import datetime


def readDict(location):
    ans = {}
    R = open(location, 'r')
    R.seek(0)
    for line in itertools.islice(R, 5, 6):
        Headers = line.split(',')
        Headers[-1] = Headers[-1].replace('\n', '')
        for header in Headers:
            ans[header] = []
    for line in itertools.islice(R, 0, None):
        lineLis = line.split(',')
        for i in range(len(Headers)):
            ans[Headers[i]].append(float(lineLis[i]))
    for key in ans:
        ans[key] = np.array(ans[key])
    return ans

pathlist = ['.\Data\CSV\Output\Day2']
allDat = {}
for path in pathlist:
    for filename in glob.glob(os.path.join(path, '*.csv')):
        key = filename.split('\\')[-1].replace('.csv', '')
        allDat[key] = readDict(filename)

descript_lis = []
for tests in allDat:
    location = tests.split('Stream_')[0]
    timeLis = tests.split('Stream_')[1].split('_')
    dateLis = timeLis[0].split('-')
    dateStr = '%s/%s/%s_%s:%s:%s' %(dateLis[0], dateLis[1], dateLis[2], timeLis[1], timeLis[2], timeLis[3])
    Epoch_UTC = time.mktime(datetime.datetime.strptime(dateStr, "%Y/%m/%d_%H:%M:%S").timetuple())
    descript_lis.append((location, Epoch_UTC))