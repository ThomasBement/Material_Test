# ---------------------------------------- #
# TestInterface [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-19
# ---------------------------------------- #
"""
IMPORTS
"""
from serial import Serial
from drawnow import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

"""
PLOTTING AND DATA COLLECTION
"""
comPort = 'COM5'#input('Enter your COM port:')
filName = input('Enter file name to save output as: ')

style.use('fivethirtyeight')

# Define figure and title
fig = plt.figure()
fig.suptitle('Time series plot:')
# Define subplots
ax1 = fig.add_subplot(1,1,1)
dat = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
ArduinoData = Serial(comPort, baudrate=9600)

def moistCal(val):
    return ((1/(880 - 1023))*(val-1023))

def animate(i):
    ArduinoString = ArduinoData.readline().decode('ASCII').replace('\n', '')
    FloatList = [float(elem) for elem in ArduinoString.split(' ') if elem != '']
    if (len(FloatList) == 13):
        dat[0].append(0.001*FloatList[0])
        dat[1].append(FloatList[1])
        dat[2].append(FloatList[2])
        dat[3].append(FloatList[3])
        dat[4].append(FloatList[4])
        dat[5].append(FloatList[5])
        dat[6].append(FloatList[6])
        dat[7].append(FloatList[7])
        dat[8].append(FloatList[8])
        dat[9].append(FloatList[9])
        dat[10].append(FloatList[10])
        dat[11].append(FloatList[11])
        dat[12].append(FloatList[12])


    plt.subplots_adjust(hspace=0.6)
    ax1.clear()
    ax1.title.set_text('Particles Under 0.3um vs. Time')
    ax1.set(xlabel='Time (s)', ylabel='Particle Count')
    if (len(dat[0]) <= 6):
        ax1.plot(dat[0], dat[1])
    else:
        ax1.plot(dat[0][-6:-1], dat[1][-6:-1]) 
    

start_time = time.time()
ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()
ArduinoData.close()
plt.close()
f = open("%s.csv" %filName,"w+")
f.write('Epoch_UTC,0.3_Bin,0.5_Bin,1.0_Bin,2.5_Bin,5.0_Bin,10.0_Bin,PM_1.0_SP,PM_2.5_SP,PM_10.0_SP,PM_1.0_AM,PM_2.5_AM,PM_10.0_AM\n')
for i in range(len(dat[0])-1):
    f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' %((dat[0][i]+start_time),dat[1][i],dat[2][i],dat[3][i],dat[4][i],dat[5][i],dat[6][i],dat[7][i],dat[8][i],dat[9][i],dat[10][i],dat[11][i],dat[12][i]))
f.close()