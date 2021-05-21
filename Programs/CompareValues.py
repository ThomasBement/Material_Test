# ---------------------------------------- #
# CompareValues [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-18
# ---------------------------------------- #

import matplotlib.pyplot as plt

def compareArr(arrX1, arrY1, arrY2, boxSz = 5):
    if (len(arrY1) != len(arrY2)):
        print('Array sized do not match')
        quit()
    else:
        arrComp = []
        arrX = []
        for i in range(boxSz, len(arrY1)):
            avgY = [0, 0]
            avgX = 0
            for j in range(boxSz):
                avgY[0] += arrY1[i-j]/boxSz
                avgY[1] += arrY2[i-j]/boxSz
                avgX += arrX1[i-j]/boxSz
            arrX.append(avgX)
            arrComp.append(abs(avgY[0]-avgY[1]))
        return arrX, arrComp

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y1 = [5, 9, 2, 3, 7, 15, 25, 70, 73, 71, 40, 22, 16, 3, 2, 1, 2, 8, 9, 6, 2]
y2 = []
for i in range(len(y1)):
    y2.append(y1[i]+15)
y2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
x2, y3 = compareArr(x, y1, y2, 4)
plt.plot(x2, y3)
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
plt.close()
