# ---------------------------------------- #
# RunTest [Python File]
# Written By: Thomas Bement
# Created On: 2021-05-14
# ---------------------------------------- #

from pynput.keyboard import Key, Listener, Controller
import time
from time import sleep

starts = []
stops = []
privious = ['Stop']
keyboard = Controller()

def on_release(key):
    if (key == Key.shift_r):
        if (privious[-1] == 'Start'):
            stops.append(time.time())
            privious.append('Stop')
        else:
            print('Invalid entry, you should be starting a data colection mesurement...')
    if (key == Key.shift_l):
        if (privious[-1] == 'Stop'):
            starts.append(time.time())
            privious.append('Start')
        else:
            print('Invalid entry, you should be stopping a data colection mesurement...')
    elif (key == Key.esc):
        return False
    else:
        if (privious[-1] == 'Stop'):
            print('Started test at: %s' %str(time.time()), end="\r")
        elif (privious[-1] == 'Start'):
            print('Stopped test at: %s' %str(time.time()), end="\r")
    
# Get user input
savePath = input('Enter Save Path: ')
saveFil = input('Enter Save File Name: ')


# Collect events until released
with Listener(on_release=on_release) as listener:
    listener.join()

# Write output times
W = open('%s\%s.csv' %(savePath, saveFil), 'w')
W.write('Start_Time,Stop,Time\n')
for i in range(len(starts)):
    W.write('%s,%s\n' %(str(starts[i]),str(stops[i])))
W.close()