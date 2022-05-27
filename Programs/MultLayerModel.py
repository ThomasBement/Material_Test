# ---------------------------------------- #
# MultLayerModel [Python File]
# Written By: Thomas Bement
# Created On: 2021-07-22
# ---------------------------------------- #

import math
import matplotlib.ticker

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Trys to convert to float, returns true if string can be converted
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def coeff_determ(y, y_fit):
    y = np.array(y)
    y_fit = np.array(y_fit)
    fitted = []
    for i in range(len(y)):
        fitted.append(find_nearest(y_fit, y[i]))
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)


# Plots penitration vs. Optical Particle Size for samples listed
def penitration_plt(allDat, bins, samples):
    legLis = []
    for i in range(len(samples)):
        idx = np.where(allDat['Sample'] == samples[i])
        legLis.append('%sx%s %s' %(allDat['Filter'][idx][0], allDat['Layers'][idx][0], allDat['Date'][idx][0]))
        plt_dat = [[], []]
        for key in allDat:
            if key in bins:
                plt_dat[0].append(float(key))
                plt_dat[1].append(allDat[key][idx])
        plt.plot(plt_dat[0], plt_dat[1])

    plt.title('Penetration vs. Optical Particle Size')
    plt.xlabel('Optical Particle Size [um]')
    plt.ylabel('Penetration [%]')
    plt.legend(legLis, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.close()

# Model for multilayer stacks
def layer_model(layers, a):
    return 1 - (np.exp(-a*layers)) #a*(b + (layers**c))

def layer_model_press(layers, a):
    return a*layers

# Plots filtration efficency vs. number of layers
"""
def layer_plt(allDat, filter, voltage='N'):
    def single_bin_plt(allDat, filter, param, colors, voltage='N'):
        idx_lis = np.where(allDat['Filter'] == filter)
        layer_dat = []
        eff_dat = []

        for idx in idx_lis[0]:
            if (allDat['Voltage'][idx] == voltage):
                layer_dat.append(allDat['Layers'][idx])
                eff_dat.append(1 - allDat[param][idx])

        # Get unique layer values to average other data with
        layers = [[], []]
        for i in range(len(layer_dat)):
            if layer_dat[i] not in layers[0]:
                layers[0].append(layer_dat[i])
                layers[1].append(i)

        plt_dat = [[0], [0], [0.0001]] # Error for 0, 0 is needed for curve fit to work
        for i in range(1, len(layers[1]) + 1):
            if i == len(layers[1]):
                plt_dat[0].append(np.mean(np.array(layer_dat[layers[1][i-1]:])))
                plt_dat[1].append(np.mean(np.array(eff_dat[layers[1][i-1]:])))
                plt_dat[2].append(np.std(np.array(eff_dat[layers[1][i-1]:])))
            else:
                plt_dat[0].append(np.mean(np.array(layer_dat[layers[1][i-1]:layers[1][i]])))
                plt_dat[1].append(np.mean(np.array(eff_dat[layers[1][i-1]:layers[1][i]])))
                plt_dat[2].append(np.std(np.array(eff_dat[layers[1][i-1]:layers[1][i]])))

        # Perform curve fit to apply theoretical model to data
        p0 = [1.5]
        popt, pcov = curve_fit(layer_model, plt_dat[0], plt_dat[1], p0=p0, sigma=plt_dat[2], absolute_sigma=True)
        x_fit = np.linspace(min(plt_dat[0]), max(plt_dat[0]), 64)
        y_fit = layer_model(x_fit, *popt)
        a = popt
        
        fit_eq = 'Model: 1-(e^-%.4f*n)' %(a)
        R_2 = 'R^2: %.4f' %(coeff_determ(plt_dat[1], y_fit))

        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height

        plt.errorbar(plt_dat[0], plt_dat[1], yerr=plt_dat[2], color=colors[0], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
        return (param, x_fit, y_fit, colors[1], fit_eq, R_2)

    bin_lis = ['0.3', '0.57', '1.09', '1.68', '2.08', '2.58']
    cmap1 = get_cmap(len(bin_lis) + 1)
    cmap2 = get_cmap(len(bin_lis) + 1, 'plasma')

    fit_dat = []
    for i in range(len(bin_lis)):
        param, x_fit, y_fit, color, fit_eq, R_2 = single_bin_plt(allDat, filter, bin_lis[i], [cmap1(i), cmap2(i)], voltage)
        fit_dat.append((param, x_fit, y_fit, color, fit_eq, R_2))
    for i in range(len(fit_dat)-1, -1, -1):
        plt.plot(fit_dat[i][1], fit_dat[i][2], label='D_opt: %s, %s, %s' %(fit_dat[i][0], fit_dat[i][4], fit_dat[i][5]), color=fit_dat[i][3])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Number of Layers [#]')
    plt.ylabel('Filtration Efficiency')
    if (voltage == 'N'):
        plt.title('Filtration Efficiency vs. Number of Layers for %s with Voltage Off' %(filter))
        plt.savefig('.\Data\Figures\Pen_VS_Layers\%s_Layers_VOFF.png' %(filter), format='png', bbox_inches='tight')
    elif (voltage == 'Y'):
        plt.title('Filtration Efficiency vs. Number of Layers for %s with Voltage On' %(filter))
        plt.savefig('.\Data\Figures\Pen_VS_Layers\%s_Layers_VON.png' %(filter), format='png', bbox_inches='tight')
    plt.show()
    plt.close()
"""
def single_bin_plt(allDat, filter, param, colors, voltage='N', x_rng=None):
    idx_lis = np.where(allDat['Filter'] == filter)
    layer_dat = []
    eff_dat = []

    for idx in idx_lis[0]:
        if (allDat['Voltage'][idx] == voltage):
            layer_dat.append(allDat['Layers'][idx])
            eff_dat.append(1 - allDat[param][idx])

    # Get unique layer values to average other data with
    layers = [[], []]
    for i in range(len(layer_dat)):
        if layer_dat[i] not in layers[0]:
            layers[0].append(layer_dat[i])
            layers[1].append(i)

    plt_dat = [[0], [0], [0.0001]] # Error for 0, 0 is needed for curve fit to work
    for i in range(1, len(layers[1]) + 1):
        if i == len(layers[1]):
            plt_dat[0].append(np.mean(np.array(layer_dat[layers[1][i-1]:])))
            plt_dat[1].append(np.mean(np.array(eff_dat[layers[1][i-1]:])))
            plt_dat[2].append(np.std(np.array(eff_dat[layers[1][i-1]:]))/np.sqrt(3))
        else:
            plt_dat[0].append(np.mean(np.array(layer_dat[layers[1][i-1]:layers[1][i]])))
            plt_dat[1].append(np.mean(np.array(eff_dat[layers[1][i-1]:layers[1][i]])))
            plt_dat[2].append(np.std(np.array(eff_dat[layers[1][i-1]:layers[1][i]]))/np.sqrt(3))
    
    #print(filter, plt_dat[0], plt_dat[1], plt_dat[2])
    # Perform curve fit to apply theoretical model to data
    p0 = [1.5]
    popt, pcov = curve_fit(layer_model, plt_dat[0], plt_dat[1], p0=p0, sigma=plt_dat[2], absolute_sigma=True)
    if (x_rng == None):
        x_fit = np.linspace(min(plt_dat[0]), max(plt_dat[0]), 64)
    else:
        x_fit = np.linspace(x_rng[0], x_rng[1], 64)
    y_fit = layer_model(x_fit, *popt)
    a = popt
    
    fit_eq = 'Model: 1-(e^-%.4f*n)' %(a)
    R_2 = 'R^2: %.4f' %(coeff_determ(plt_dat[1], y_fit))

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    plt.errorbar(plt_dat[0], plt_dat[1], yerr=plt_dat[2], color=colors[0], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
    return (param, x_fit, y_fit, colors[1], fit_eq, R_2)

def layer_plt(allDat, filter, voltage='N'):
    bin_lis = ['0.3', '0.57', '1.09', '1.68', '2.08', '2.58']
    cmap1 = get_cmap(len(bin_lis) + 1)
    cmap2 = get_cmap(len(bin_lis) + 1, 'plasma')

    fit_dat = []
    for i in range(len(bin_lis)):
        param, x_fit, y_fit, color, fit_eq, R_2 = single_bin_plt(allDat, filter, bin_lis[i], [cmap1(i), cmap2(i)], voltage)
        fit_dat.append((param, x_fit, y_fit, color, fit_eq, R_2))
    for i in range(len(fit_dat)-1, -1, -1):
        plt.plot(fit_dat[i][1], fit_dat[i][2], label='D_opt: %s, %s, %s' %(fit_dat[i][0], fit_dat[i][4], fit_dat[i][5]), color=fit_dat[i][3])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Number of Layers [#]')
    plt.ylabel('Filtration Efficiency')
    if (voltage == 'N'):
        plt.title('Filtration Efficiency vs. Number of Layers for %s with Voltage Off' %(filter))
        plt.savefig('.\Data\Figures\Pen_VS_Layers\%s_Layers_VOFF.png' %(filter), format='png', bbox_inches='tight')
    elif (voltage == 'Y'):
        plt.title('Filtration Efficiency vs. Number of Layers for %s with Voltage On' %(filter))
        plt.savefig('.\Data\Figures\Pen_VS_Layers\%s_Layers_VON.png' %(filter), format='png', bbox_inches='tight')
    plt.show()
    plt.close()

def layer_plt_masks(allDat, mask_lis, size, voltage='N', x_rng=None):
    cmap1 = get_cmap(len(mask_lis) + 1, 'plasma')
    cmap2 = get_cmap(len(mask_lis) + 1, 'plasma')

    fit_dat = []
    for i in range(len(mask_lis)):
        param, x_fit, y_fit, color, fit_eq, R_2 = single_bin_plt(allDat, mask_lis[i], size, [cmap1(i), cmap2(i)], voltage, x_rng)
        print(' %s, %s, %s' %(mask_lis[i], fit_eq, R_2))
        fit_dat.append((mask_lis[i], x_fit, y_fit, color, fit_eq, R_2))

    for i in range(len(fit_dat)-1, -1, -1):
        plt.plot(fit_dat[i][1], fit_dat[i][2], label=' %s, %s, %s' %(fit_dat[i][0], fit_dat[i][4], fit_dat[i][5]), color=fit_dat[i][3])
        #print(' %s, %s, %s' %(fit_dat[i][0], fit_dat[i][4], fit_dat[i][5]))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Number of Layers [#]')
    plt.ylabel('Filtration Efficiency')
    if (voltage == 'N'):
        #plt.title('Filtration Efficiency vs. Number of Layers at %s um with Voltage Off' %(size))
        plt.savefig('.\Data\Figures\Pen_VS_Layers\Masks_VOFF_%s.svg' %(size), format='svg', bbox_inches='tight')
    elif (voltage == 'Y'):
        #plt.title('Filtration Efficiency vs. Number of Layers at %s um with Voltage On' %(size))
        plt.savefig('.\Data\Figures\Pen_VS_Layers\Masks_VON_%s.svg' %(size), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    #print(fit_dat)

def layer_plt_masks_press(allDat, mask_lis, voltage='N', x_rng=None):
    cmap1 = get_cmap(len(mask_lis) + 1, 'plasma')
    cmap2 = get_cmap(len(mask_lis) + 1, 'plasma')

    fit_dat = []
    for i in range(len(mask_lis)):
        layer_plt_press(allDat, mask_lis[i], [cmap1(i), cmap2(i)])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Number of Layers [#]')
    plt.ylabel('Pressure Drop [Pa]')
    if (voltage == 'N'):
        plt.title('Pressure Drop vs. Number of Layers with Voltage Off')
        plt.savefig('.\Data\Figures\Press_VS_Layers\Masks_VOFF_Press.svg', format='svg', bbox_inches='tight')
    elif (voltage == 'Y'):
        plt.title('Pressure Drop vs. Number of Layers with Voltage On')
        plt.savefig('.\Data\Figures\Press_VS_Layers\Masks_VON_Press.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def layer_plt_press(allDat, filter, colors, voltage='N'):
    idx_lis = np.where(allDat['Filter'] == filter)
    layer_dat = []
    eff_dat = []

    for idx in idx_lis[0]:
        if (allDat['Voltage'][idx] == voltage):
            layer_dat.append(allDat['Layers'][idx])
            eff_dat.append(allDat['dP'][idx])

    # Get unique layer values to average other data with
    layers = [[], []]
    for i in range(len(layer_dat)):
        if layer_dat[i] not in layers[0]:
            layers[0].append(layer_dat[i])
            layers[1].append(i)

    plt_dat = [[0], [0], [0.0001]] # Error for 0, 0 is needed for curve fit to work
    for i in range(1, len(layers[1]) + 1):
        if i == len(layers[1]):
            plt_dat[0].append(np.mean(np.array(layer_dat[layers[1][i-1]:])))
            plt_dat[1].append(np.mean(np.array(eff_dat[layers[1][i-1]:])))
            plt_dat[2].append(np.std(np.array(eff_dat[layers[1][i-1]:]))/np.sqrt(3))
        else:
            plt_dat[0].append(np.mean(np.array(layer_dat[layers[1][i-1]:layers[1][i]])))
            plt_dat[1].append(np.mean(np.array(eff_dat[layers[1][i-1]:layers[1][i]])))
            plt_dat[2].append(np.std(np.array(eff_dat[layers[1][i-1]:layers[1][i]]))/np.sqrt(3))

    #print(filter, plt_dat[0], plt_dat[1], plt_dat[2])
    # Perform curve fit to apply theoretical model to data
    p0 = [1]
    popt, pcov = curve_fit(layer_model_press, plt_dat[0], plt_dat[1], p0=p0, sigma=plt_dat[2], absolute_sigma=True)
    x_fit = np.linspace(min(plt_dat[0]), max(plt_dat[0]), 64)
    y_fit = layer_model_press(x_fit, *popt)
    a = popt[0]
    
    fit_eq = 'Model: %.4f*n' %(a)
    R_2 = 'R^2: %.4f' %(coeff_determ(plt_dat[1], y_fit))
    print(' %s, %s, %s' %(filter, fit_eq, R_2))

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    plt.errorbar(plt_dat[0], plt_dat[1], yerr=plt_dat[2], color=colors[0], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
    plt.plot(x_fit, y_fit, color=colors[0], label=' %s, %s, %s' %(filter, fit_eq, R_2))

def pen_plt(allDat, filter, layers, voltage='N'):
    bins, layer_dat, pen_dat = [], [], []
    for key in allDat:
        if '.' in key:
            bins.append(key)
            pen_dat.append([])

    idx_lis_filt = np.where(allDat['Filter'] == filter)
    for idx in idx_lis_filt[0]:
        if (allDat['Voltage'][idx] == voltage): 
            layer_dat.append(allDat['Layers'][idx])
            for i, key in enumerate(bins):
                pen_dat[i].append(allDat[key][idx])

    idx_lis_lay = np.where(np.array(layer_dat) == layers)

    plt_dat_sct = [bins]
    for i in range(len(idx_lis_lay[0])):
        plt_dat_sct.append([])
    for i in range(len(pen_dat)):
        for j in range(len(idx_lis_lay[0])):
            plt_dat_sct[j+1].append(pen_dat[i][idx_lis_lay[0][j]])

    for i in range(1, len(plt_dat_sct)):
        plt.scatter(plt_dat_sct[0], plt_dat_sct[i])
    #plt.xscale('log', base = 10) 
    plt.xticks(rotation = 90)
    plt.xlabel('Optical Particle Size [um]')
    plt.ylabel('Particle Penetration')
    if (voltage == 'N'):
        plt.title('Particle Penetration vs. Optical Particle Size for %s_%s_Ply with Voltage Off' %(filter, layers))
        plt.savefig('.\Data\Figures\Pen_VS_Size\Scatter\%s_%s_Ply_VOFF_%s.png' %(filter, layers, param), format='png', bbox_inches='tight')
    elif (voltage == 'Y'):
        plt.title('Particle Penetration vs. Optical Particle Size for %s_%s_Ply with Voltage On' %(filter, layers))
        plt.savefig('.\Data\Figures\Pen_VS_Size\Scatter\%s_%s_Ply_VON_%s.png' %(filter, layers, param), format='png', bbox_inches='tight')
    plt.show()
    plt.close()

    plt_dat_err = [bins, [], []]
    for i in range(len(pen_dat)):
        start = idx_lis_lay[0][0]
        stop = idx_lis_lay[0][-1] + 1
        plt_dat_err[1].append(np.mean(np.array(pen_dat[i][start:stop])))
        plt_dat_err[2].append(np.std(np.array(pen_dat[i][start:stop]))/np.sqrt(3))
    
    plt.errorbar(plt_dat_err[0], plt_dat_err[1], yerr=plt_dat_err[2], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
    #plt.xscale('log', base = 10) 
    plt.xticks(rotation = 90)
    plt.xlabel('Optical Particle Size [um]')
    plt.ylabel('Particle Penetration')
    if (voltage == 'N'):
        plt.title('Particle Penetration vs. Optical Particle Size for %s_%s_Ply with Voltage Off' %(filter, layers))
        plt.savefig('.\Data\Figures\Pen_VS_Size\Errorbar\%s_%s_Ply_VOFF.png' %(filter, layers), format='png', bbox_inches='tight')
    if (voltage == 'Y'):
        plt.title('Particle Penetration vs. Optical Particle Size for %s_%s_Ply with Voltage On' %(filter, layers))
        plt.savefig('.\Data\Figures\Pen_VS_Size\Errorbar\%s_%s_Ply_VON.png' %(filter, layers), format='png', bbox_inches='tight')
    plt.show()
    plt.close()

def pen_plt_overlay(allDat, filter1, filter2, voltage1, voltage2, layers):
    def error_plt(allDat, filter, layers, voltage):
        bins, layer_dat, pen_dat = [], [], []
        for key in allDat:
            if '.' in key:
                bins.append(key)
                pen_dat.append([])
        
        idx_lis_filt = np.where(allDat['Filter'] == filter)
        for idx in idx_lis_filt[0]:
            if (allDat['Voltage'][idx] == voltage):
                layer_dat.append(allDat['Layers'][idx])
                for i, key in enumerate(bins):
                    pen_dat[i].append(allDat[key][idx])

        idx_lis_lay = np.where(np.array(layer_dat) == layers)
        plt_dat_err = [bins, [], []]
        for i in range(len(pen_dat)):
            start = idx_lis_lay[0][0]
            stop = idx_lis_lay[0][-1] + 1
            plt_dat_err[1].append(np.mean(np.array(pen_dat[i][start:stop], dtype='float')))
            plt_dat_err[2].append(np.std(np.array(pen_dat[i][start:stop], dtype='float')))
        
        plt.errorbar(plt_dat_err[0], plt_dat_err[1], yerr=plt_dat_err[2], marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
        return plt_dat_err[0], plt_dat_err[1], plt_dat_err[2], pen_dat
        #plt.xscale('log', base = 10) 

    dat1_x, dat1, dat1_err, pen_dat1 = error_plt(allDat, filter1, layers, voltage1)
    dat2_x, dat2, dat2_err, pen_dat2 = error_plt(allDat, filter2, layers, voltage2)
        
    plt.xticks(rotation = 90)
    vlt1_str, vlt2_str = 'OFF', 'OFF'
    if voltage1 == 'Y':
        vlt1_str = 'ON'
    if voltage2 == 'Y':
        vlt2_str = 'ON'
    plt.title('Particle Penetration vs. Optical Particle Size for %s Voltage %s and %s Voltage %s for %s Layers' %(filter1, vlt1_str, filter2, vlt2_str, layers[0]))
    plt.xlabel('Optical Particle Size [um]')
    plt.ylabel('Particle Penetration')
    plt.legend(['%s, V_%s' %(filter1, vlt1_str), '%s, V_%s' %(filter2, vlt2_str)])
    plt.savefig('.\Data\Figures\Pen_VS_Size\Overlay\%s_V%s_and_%s_V%s_%s_Ply.png' %(filter1, vlt1_str, filter2, vlt2_str, layers), format='png', bbox_inches='tight')
    plt.show()
    plt.close()

    dat3, dat3_err = [], []
    for i in range(len(pen_dat1)):
        temp = []
        for j in range(len(pen_dat1[i])):
            pen_dat1[i][j] = 1 - pen_dat1[i][j]
            pen_dat2[i][j] = 1 - pen_dat2[i][j]
            temp.append(abs(pen_dat1[i][j]-pen_dat2[i][j])/pen_dat1[i][j])
        eff_mean = np.mean(np.array(temp))
        eff_std = np.std(np.array(temp))
        dat3.append(eff_mean)
        dat3_err.append(eff_std)
    plt.errorbar(dat1_x, dat3, yerr=dat3_err, marker='o', ls='solid', capsize=3, elinewidth=1, markeredgewidth=2)
    plt.title('Change in Particle Penitration vs. Optical Particle Size for %s and %s ' %(filter1, filter2))
    plt.xlabel('Optical Particle Size [um]')
    condition1, condition2 = 'uw', 'w'
    if (' WASH' in filter2):
        condition1, condition2 = 'w', 'uw'
    plt.ylabel('Change in Particle Penitration, |Pen_%s-Pen_%s|/Pen_%s' %(condition1, condition2, condition1))
    plt.savefig('.\Data\Figures\Pen_VS_Size\Overlay\%s_and_%s_RChange.png' %(filter1, filter2), format='png', bbox_inches='tight')
    plt.show()
    plt.close()

def get_cmap(n, name='plasma'):
    return plt.cm.get_cmap(name, n)

def qual_eff(Q, dP):
    return 1- math.exp((Q*dP)/-1000)

def qual_plot_wash(allDat, size, matLis):
    # Write bins to list for future use
    bin_lis = []
    for key in allDat:
        if ('.' in key):
            bin_lis.append(key)
    # Fill dictonary using key information in tuple as key and containing efficency data
    dat_tup = {}
    for i, sample in enumerate(allDat['Sample']):
        if (allDat['Filter'][i] in matLis) and (allDat['Voltage'][i] == 'N'):
            if (allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i]) in dat_tup:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['press'].append(allDat['dP'][i])
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['eff'].append(1 - allDat[str(size)][i])
            else:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])] = {'press': [allDat['dP'][i]], 'eff': [1 - allDat[str(size)][i]]}
    print(dat_tup)
    # Generate statistics and make plot
    qual_lis = [0.5, 2, 5, 10, 20, 50, 100, 200, 500]
    press_max = 135
    eff_max = 1.01
    x_rng = np.linspace(0, press_max, 256)
    y_rng = np.ones_like(x_rng)
    filter_lis = []
    for key in dat_tup:
        if (key[0] not in filter_lis):
            filter_lis.append(key[0])
    half_plot_len = int(len(filter_lis)/2)

    cmap1 = get_cmap(half_plot_len, 'winter')
    cmap2 = get_cmap(len(filter_lis) - half_plot_len, 'autumn')
    for i, key in enumerate(dat_tup):
        press = np.array(dat_tup[key]['press'])
        eff = np.array(dat_tup[key]['eff'])
        color_idx = np.where(np.array(filter_lis) == key[0])[0][0]
        # Uncomment to output for table
        '''
        try:
            print(key[0], np.mean(eff), np.mean(press), -1000*math.log(1-np.mean(eff))/np.mean(press))
        except:
            print(key[0], np.mean(eff), np.mean(press), 'nan')
        '''
        if ('WASH' in key[0]):
            plt.errorbar(np.mean(press), np.mean(eff), xerr=None, yerr=None, 
                        marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2, color=cmap1(color_idx - half_plot_len),
                        label='%s' %(key[0]))

            # Efficency Error
            #plt.scatter([np.mean(press), np.mean(press)], [max(eff), min(eff)], marker="_", s=15, color=cmap1(color_idx))
            #plt.plot([np.mean(press), np.mean(press)], [max(eff), min(eff)], lw=0.5, color=cmap1(color_idx))
            # Pressure Error  
            #plt.scatter([max(press), min(press)], [np.mean(eff), np.mean(eff)], marker="|", s=15, color=cmap1(color_idx))
            #plt.plot([max(press), min(press)], [np.mean(eff), np.mean(eff)], lw=0.5, color=cmap1(color_idx))

        else:
            plt.errorbar(np.mean(press), np.mean(eff), xerr=None, yerr=None, 
                        marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2, color=cmap2(color_idx),
                        label='%s' %(key[0]))

            # Efficency Error
            #plt.scatter([np.mean(press), np.mean(press)], [max(eff), min(eff)], marker="_", s=15, color=cmap1(color_idx))
            #lt.plot([np.mean(press), np.mean(press)], [max(eff), min(eff)], lw=0.5, color=cmap1(color_idx))
            # Pressure Error  
            #plt.scatter([max(press), min(press)], [np.mean(eff), np.mean(eff)], marker="|", s=15, color=cmap1(color_idx))
            #plt.plot([max(press), min(press)], [np.mean(eff), np.mean(eff)], lw=0.5, color=cmap1(color_idx))

            key_wash = ('%s WASH' %(key[0]), key[1], key[2], key[3])
            press_wash = np.array(dat_tup[key_wash]['press'])
            eff_wash = np.array(dat_tup[key_wash]['eff'])
            plt.plot([np.mean(press), np.mean(press_wash)], [np.mean(eff), np.mean(eff_wash)], color=cmap2(color_idx))
    
    for Q in qual_lis:
        for i in range(len(y_rng)):
            y_rng[i] = qual_eff(Q, x_rng[i])
        plt.plot(x_rng, y_rng, linestyle='dotted', color='#858585', alpha=0.5)
    plt.title('Filtration Efficency vs. Pressure Drop at %s um' %(size))
    plt.xlabel('Pressure Drop [Pa]')
    plt.ylabel('Filtration Efficency')
    plt.xlim([35, press_max])
    plt.ylim([0.7, eff_max])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('.\Data\Figures\Qual\Qual_Plot_Masks_Wash_%s.svg' %(size), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()    

    # Generate efficency bar plots at given size
    mask_lis = []
    wash_lis = []
    for key in dat_tup:
        if (key[0].replace(' WASH', '') not in mask_lis):
            mask_lis.append(key[0].replace(' WASH', ''))
        if (' WASH' in key[0]):
            wash_lis.append(key[0])
    
    x_pos_unwashed = []
    x_pos_washed = []
    x_ticks = []
    for i in range(1, len(mask_lis)*3, 3):
        x_pos_unwashed.append(i)
        x_pos_washed.append(i+1)
    for i in range(len(x_pos_unwashed)):
        x_ticks.append((float(x_pos_unwashed[i])+float(x_pos_washed[i]))/2)


    unwashed_mean, unwashed_std = [], []
    washed_mean, washed_std = [], []
    for mask in mask_lis:
        for key in dat_tup:
            if (mask == key[0]):
                unwashed_mean.append(np.mean(np.array(dat_tup[key]['eff'])))
                unwashed_std.append(np.std(np.array(dat_tup[key]['eff'])))
                
    for mask in wash_lis:  
        for key in dat_tup:  
            if (mask == key[0]):
                washed_mean.append(np.mean(np.array(dat_tup[key]['eff'])))
                washed_std.append(np.std(np.array(dat_tup[key]['eff'])))
    plt.scatter(float('nan'), float('nan'), alpha=0.5, color='red')
    plt.scatter(float('nan'), float('nan'), alpha=0.5, color='blue')
    plt.legend(['Unwashed', 'Washed'], bbox_to_anchor=(1.05, 1), loc='upper left' )
    plt.bar(x_pos_unwashed, unwashed_mean, yerr=unwashed_std, align='center', alpha=0.5, color='red', ecolor='black', capsize=10)
    plt.bar(x_pos_washed, washed_mean, yerr=washed_std, align='center', alpha=0.5, color='blue', ecolor='black', capsize=10)
    plt.xticks(ticks=x_ticks, labels=mask_lis, rotation=0)
    plt.title('Filtration Efficiency at %.2f um for Washed and Unwashed Mask Samples' %(size))
    plt.ylabel('Filtration Efficiency')
    plt.savefig('.\Data\Figures\Qual\Eff_Masks_Wash_%f.png' %(size), format='png', bbox_inches='tight')
    plt.show()
    plt.close()

    # T-test for washed bar plot
    t_reject = 2.132
    print('Results of T-Test:\n')
    print('Mask,t,DOF,Condition')
    for mask in mask_lis:
        for key in dat_tup:
            if (mask == key[0]):
                x1 = float(np.mean(np.array(dat_tup[key]['eff'])))
                s1 = float(np.std(np.array(dat_tup[key]['eff'])))

                wash_key = ('%s WASH' %(key[0]), key[1], key[2], key[3])
                x2 = float(np.mean(np.array(dat_tup[wash_key]['eff'])))
                s2 = float(np.std(np.array(dat_tup[wash_key]['eff'])))

                n1 = float(len(dat_tup[key]['eff']))
                n2 = float(len(dat_tup[wash_key]['eff']))

                Sp = math.sqrt( (((n1 - 1)*(s1)**2) + ((n2 - 1)*(s2)**2))/(n1 + n2 - 2) )
                t = (x1 - x2)/(Sp*math.sqrt((1/n1) + (1/n2)))

                codition = 'Fail to Reject'
                if (abs(t) > t_reject):
                    codition = 'Reject'

                print('%s,%.4f,%.4f,%s' %(mask, t, (n1 + n2 - 2), codition))

def qual_plot_volt(allDat, size, matLis, mat_lis):
    # Fill dictonary using key information in tuple as key and containing efficency data
    dat_tup = {}
    for i, sample in enumerate(allDat['Sample']):
        if allDat['Filter'][i] in matLis:
            if (allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i]) in dat_tup:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['press'].append(allDat['dP'][i])
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['eff'].append(1 - allDat[str(size)][i])
            else:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])] = {'press': [allDat['dP'][i]], 'eff': [1 - allDat[str(size)][i]]}
    

    # Get voltage on and voltage off pairs
    key_lis = []
    for mat in matLis:
        for key in dat_tup:
            if (mat == key[0]):
                if ((key[0], key[1], 'Y', key[3]) not in key_lis) and ((key[0], key[1], 'N', key[3]) not in key_lis):
                    key_lis.append(key)

    # Generate efficency bar plots at given size 
    for mat in matLis:
        label_lis = []

        no_volt_mean, no_volt_std = [], []
        volt_mean, volt_std = [], []
        for key in key_lis:
            if (key[0] == mat):
                label_lis.append('%sx%i' %(key[0], key[1]))

                no_volt_key = (key[0], key[1], 'N', key[3]) 
                no_volt_mean.append(np.mean(np.array(dat_tup[no_volt_key]['press'])))
                no_volt_std.append(np.std(np.array(dat_tup[no_volt_key]['press'])))

                volt_key = (key[0], key[1], 'Y', key[3]) 
                volt_mean.append(np.mean(np.array(dat_tup[volt_key]['press'])))
                volt_std.append(np.std(np.array(dat_tup[volt_key]['press'])))
        
        x_pos_no_volt, x_pos_volt, x_ticks = [], [], []
        for i in range(1, len(no_volt_mean)*3, 3):
            x_pos_no_volt.append(i)
            x_pos_volt.append(i+1)
            x_ticks.append((2*i+1)/2)

        plt.scatter(float('nan'), float('nan'), alpha=0.5, color='red')
        plt.scatter(float('nan'), float('nan'), alpha=0.5, color='blue')
        plt.legend(['Voltage Off', 'Voltage On'], bbox_to_anchor=(1.05, 1), loc='upper left' )
        plt.bar(x_pos_no_volt, no_volt_mean, yerr=no_volt_std, align='center', alpha=0.5, color='red', ecolor='black', capsize=10)
        plt.bar(x_pos_volt, volt_mean, yerr=volt_std, align='center', alpha=0.5, color='blue', ecolor='black', capsize=10)
        plt.xticks(ticks=x_ticks, labels=label_lis, rotation=0)
        #plt.title('Pressure Drop for Voltage on and Off')
        plt.ylabel('Pressure Drop [Pa]')
        plt.savefig('.\Data\Figures\Qual\Press_%s_%.2f_Volt.png' %(mat, size), format='png', bbox_inches='tight')
        #plt.show()
        plt.close()

    marker_dic = {'SB': 'o', 'SMS': '^', 'W': 'D', 'ML': 's'}
    cmap = get_cmap(len(mat_lis)+2)
    plt_dat = {}
    label_lis = []
    x_range = [0, 0]
    for key in key_lis:
        print(key)
        new_key = '%sx%i' %(key[0], key[1])
        plt_dat[new_key] = [ [[],[]], [[],[]] ]

        no_volt_key = (key[0], key[1], 'Y', key[3])
        plt_dat[new_key][0][0].append(np.mean(np.array(dat_tup[no_volt_key]['eff'])))
        plt_dat[new_key][0][1].append(np.std(np.array(dat_tup[no_volt_key]['eff']))/np.sqrt(3))
        if (x_range[1] <= plt_dat[new_key][0][0][-1]):
            x_range[1] = plt_dat[new_key][0][0][-1]

        volt_key = (key[0], key[1], 'N', key[3])
        plt_dat[new_key][1][0].append(np.mean(np.array(dat_tup[volt_key]['eff'])))
        plt_dat[new_key][1][1].append(np.std(np.array(dat_tup[volt_key]['eff']))/np.sqrt(3))

    sample_lis = []
    for i, key in enumerate(plt_dat):
        if key.split('x')[0] in matLis:
            if key.split('x')[0] not in sample_lis:
                sample_lis.append(key.split('x')[0])
    sample_lis = np.array(sample_lis)

    leg_lis = []
    for i, key in enumerate(plt_dat):
        if key.split('x')[0] in sample_lis:
            j = np.where(sample_lis == key.split('x')[0])[0][0]
        plt.errorbar(plt_dat[key][0][0], plt_dat[key][1][0], xerr=plt_dat[key][0][1], yerr=plt_dat[key][1][1], 
                    marker=marker_dic[mat_lis[j]], ls='none', capsize=3, elinewidth=1, markeredgewidth=2, color=cmap(j),
                    label='%s' %(key))
    plt.legend()  
    plt.plot(x_range, x_range, color=cmap(-2), alpha=0.6)
    #plt.title('Filtration Efficiency with Voltage On and Off at %.2f um' %(size))
    plt.xlabel('Filtration Efficiency with Voltage Off')
    plt.ylabel('Filtration Efficiency with Voltage On')
    plt.savefig('.\Data\Figures\Qual\Von_Voff_%.2f_Eff.svg' %(size), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

    # T-test for washed bar plot
    t_reject = 1.533
    print('Results of T-Test:\n')
    print('Mask,t,DOF,Condition')
    for key in key_lis:
        no_volt_key = (key[0], key[1], 'Y', key[3])
        volt_key = (key[0], key[1], 'N', key[3])

        x1 = float(np.mean(np.array(dat_tup[no_volt_key]['eff'])))
        s1 = float(np.std(np.array(dat_tup[no_volt_key]['eff'])))
        n1 = float(len(dat_tup[no_volt_key]['eff']))

        x2 = float(np.mean(np.array(dat_tup[volt_key]['eff'])))
        s2 = float(np.std(np.array(dat_tup[volt_key]['eff'])))
        n2 = float(len(dat_tup[volt_key]['eff']))

        Sp = math.sqrt( (((n1 - 1)*(s1)**2) + ((n2 - 1)*(s2)**2))/(n1 + n2 - 2) )
        t = (x1 - x2)/(Sp*math.sqrt((1/n1) + (1/n2)))

        codition = 'Fail to Reject'
        if (abs(t) > t_reject):
            codition = 'Reject'

        print('%sx%i,%.4f,%.4f,%s' %(key[0], key[1], t, (n1 + n2 - 2), codition))

    

def qual_plot(allDat, size, matLis):
    # Write bins to list for future use
    bin_lis = []
    for key in allDat:
        if ('.' in key):
            bin_lis.append(key)
    # Fill dictonary using key information in tuple as key and containing efficency data
    dat_tup = {}
    for i, sample in enumerate(allDat['Sample']):
        if (allDat['Filter'][i] in matLis) and (allDat['Voltage'][i] == 'N'):
            if (allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i]) in dat_tup:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['press'].append(allDat['dP'][i])
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['eff'].append(1 - allDat[str(size)][i])
            else:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])] = {'press': [allDat['dP'][i]], 'eff': [1 - allDat[str(size)][i]]}

    # Generate statistics and make plot
    qual_lis = [0.5, 2, 5, 10, 20, 50, 100, 200, 500]
    marker_lis = ['o', '^', 'X', 'D', '*', 'v']
    press_max = 110
    eff_max = 1.01
    x_rng = np.linspace(0, press_max, 256)
    y_rng = np.ones_like(x_rng)
    filter_lis = []
    for key in dat_tup:
        if (key[0] not in filter_lis):
            filter_lis.append(key[0])
    half_plot_len = int(len(filter_lis)/2)

    marker_dic = {'SB': 'o', 'SMS': '^', 'W': 'D', 'ML': 's'}
    mat_lis = ['SB', 'SMS', 'W', 'ML']
    cmap1 = get_cmap(len(marker_dic)+1)
    
    leg_lis = []
    for key in marker_dic:
        color_idx = np.where(np.array(mat_lis) == key)[0][0]
        plt.scatter([50], [100], marker=marker_dic[key], s=30, color=cmap1(color_idx))
        leg_lis.append(key)
    #plt.legend(leg_lis, bbox_to_anchor=(1.05, 1), loc='upper left')

    W = open('%.2f_OutPut.csv' %(size), 'w')
    W.write('Material,Layers,Press Mean,Eff Mean,Press STD,Eff STD')
    for i, key in enumerate(dat_tup):
        press = np.array(dat_tup[key]['press'])
        eff = np.array(dat_tup[key]['eff'])
        color_idx = np.where(np.array(mat_lis) == key[3])[0][0]
        #print(np.array(single_mat_lis), np.array(materl_lis)[i], i, len(materl_lis))
        # Uncomment to output for table
        """
        try:
            print(key[0], np.mean(eff), np.mean(press), -1000*math.log(1-np.mean(eff))/np.mean(press))
        except:
            print(key[0], np.mean(eff), np.mean(press), 'nan')
        """
        W.write('\n')
        W.write(','.join([str(key[0]), str(key[1]), str(np.mean(press)), str(np.mean(eff)), str(np.std(press)), str(np.std(eff))]))
        plt.errorbar(np.mean(press), np.mean(eff), xerr=None, yerr=None, 
                    marker=marker_dic[key[3]], ms=5, ls='none', capsize=2, elinewidth=1, markeredgewidth=2, label='%sx%i' %(key[0], key[1]))
        # Efficency Error
        plt.scatter([np.mean(press), np.mean(press)], [max(eff), min(eff)], marker="_", s=15, color=cmap1(color_idx))
        plt.plot([np.mean(press), np.mean(press)], [max(eff), min(eff)], lw=0.5, color=cmap1(color_idx))
        # Pressure Error  
        plt.scatter([max(press), min(press)], [np.mean(eff), np.mean(eff)], marker="|", s=15, color=cmap1(color_idx))
        plt.plot([max(press), min(press)], [np.mean(eff), np.mean(eff)], lw=0.5, color=cmap1(color_idx))

    W.close()
    for Q in qual_lis:
        for i in range(len(y_rng)):
            y_rng[i] = qual_eff(Q, x_rng[i])
        plt.plot(x_rng, y_rng, linestyle='dotted', color='#858585', alpha=0.5)
    #plt.title('Filtration Efficency vs. Pressure Drop at %s um' %(size))
    plt.xlabel('Pressure Drop [Pa]')
    plt.ylabel('Filtration Efficency')
    plt.xlim([0, press_max])
    plt.ylim([0, eff_max])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('.\Data\Figures\Qual\Qual_Plot_Method_0.962_Err.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()    

def ops_2_aero(ops_size):
    ops_lis = [0.300, 0.370, 0.460, 0.570, 0.710, 0.880, 1.090, 1.350, 1.680, 2.080, 2.580, 3.200, 3.960, 4.920, 6.100, 7.560]
    aero_lis = [0.498, 0.620, 0.796, 0.962, 1.190, 1.480, 1.910, 2.320, 2.760, 3.400, 4.220, 5.250, 6.490, 8.120, 10.000, 12.200]
    for i in range(len(ops_lis)):
        if (ops_size == ops_lis[i]):
            return aero_lis[i]

def size_wash(allDat, size_lis, matLis):
    # Convert OPS size to aerodynamic for size resolved plots
    aero_lis = []
    for i in range(len(size_lis)):
        aero_lis.append(ops_2_aero(size_lis[i]))

    # Write bins to list for future use
    bin_lis = []
    for key in allDat:
        if ('.' in key):
            bin_lis.append(key)
    # Fill dictonary using key information in tuple as key and containing efficency data
    dat_tup = {}
    for i, sample in enumerate(allDat['Sample']):
        if (allDat['Filter'][i] in matLis) and (allDat['Voltage'][i] == 'N'):
            if (allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i]) in dat_tup:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['press'].append(allDat['dP'][i])
                temp = {}
                for j in range(len(size_lis)):
                    temp[size_lis[j]] = 1 - allDat[str(size_lis[j])][i]
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['eff'].append(temp)
            else:
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])] = {'press': [allDat['dP'][i]], 'eff': []}
                temp = {}
                for j in range(len(size_lis)):
                    temp[size_lis[j]] = 1 - allDat[str(size_lis[j])][i]
                dat_tup[(allDat['Filter'][i], allDat['Layers'][i], allDat['Voltage'][i], allDat['Material'][i])]['eff'].append(temp)

    # Format plotting data for averaging
    plt_dat = {}
    for tup in dat_tup:
        temp = {}
        qual = {}
        for i in range(len(size_lis)):
            temp[size_lis[i]] = []
            qual[size_lis[i]] = []
        for i in range(len(dat_tup[tup]['eff'])):
            for j in range(len(dat_tup[tup]['eff'][i])):
                temp[size_lis[j]].append(dat_tup[tup]['eff'][i][size_lis[j]])
                qual[size_lis[j]].append(-1000*np.log(1-dat_tup[tup]['eff'][i][size_lis[j]])/dat_tup[tup]['press'])
                
        
        averaged = []
        quality = []
        for i in range(len(size_lis)):
            averaged.append(np.mean(temp[size_lis[i]]))
            quality.append(np.mean(qual[size_lis[i]]))
        plt_dat[tup] = {'press': dat_tup[tup]['press'], 'eff': averaged, 'size_ops': size_lis, 'size_aero': aero_lis, 'qual': quality}
    
    print(plt_dat)

    mask_lis = []
    for tup in plt_dat:
        if 'WASH' not in tup[0]:
            mask_lis.append(tup[0])
    cmap = get_cmap(len(mask_lis)+1)
    
    fig1, ax1 = plt.subplots()
    # Hardcoded symbols :(
    marker_lis = ['o', '^', 'D', 's', 'p', 'X', 'P']
    for tup in plt_dat:
        if 'WASH' in tup[0]:
            for i in range(len(mask_lis)):
                if mask_lis[i] in tup[0]:
                    ax1.plot(plt_dat[tup]['size_ops'], plt_dat[tup]['eff'], marker=marker_lis[i], ms=5, ls=':', color=cmap(i), label = '%s, P: %.0f' %(tup[0], np.mean(plt_dat[tup]['press'])))
        else:
            for i in range(len(mask_lis)):
                if mask_lis[i] in tup[0]:
                    ax1.plot(plt_dat[tup]['size_ops'], plt_dat[tup]['eff'], marker=marker_lis[i], ms=5, color=cmap(i), label = '%s, P: %.0f' %(tup[0], np.mean(plt_dat[tup]['press'])))
    ax1.set_xscale('log')
    ax1.set_xticks(size_lis)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xlabel('OPS Diameter [um]')
    ax1.set_ylabel('Filtration Efficency')

    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_xaxis().set_tick_params(which='minor', size=0)
    ax1.get_xaxis().set_tick_params(which='minor', width=0) 

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('.\Data\Figures\Wash_Size\Wash_Size_Res_OPS.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

    fig1, ax1 = plt.subplots()
    for tup in plt_dat:
        if 'WASH' in tup[0]:
            for i in range(len(mask_lis)):
                if mask_lis[i] in tup[0]:
                    ax1.plot(plt_dat[tup]['size_aero'], plt_dat[tup]['eff'], marker=marker_lis[i], ms=5, ls=':', color=cmap(i), label = '%s, P: %.0f' %(tup[0], np.mean(plt_dat[tup]['press'])))
        else:
            for i in range(len(mask_lis)):
                if mask_lis[i] in tup[0]:
                    ax1.plot(plt_dat[tup]['size_aero'], plt_dat[tup]['eff'], marker=marker_lis[i], ms=5, color=cmap(i), label = '%s, P: %.0f' %(tup[0], np.mean(plt_dat[tup]['press'])))
    ax1.set_xscale('log')
    ax1.set_xticks(aero_lis)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xlabel('Aerodynamic Diameter [um]')
    ax1.set_ylabel('Filtration Efficency')

    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_xaxis().set_tick_params(which='minor', size=0)
    ax1.get_xaxis().set_tick_params(which='minor', width=0)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('.\Data\Figures\Wash_Size\Wash_Size_Res_Aero.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

    ### QUALITY ###

    fig1, ax1 = plt.subplots()
    for tup in plt_dat:
        if 'WASH' in tup[0]:
            for i in range(len(mask_lis)):
                if mask_lis[i] in tup[0]:
                    ax1.plot(plt_dat[tup]['size_aero'], plt_dat[tup]['qual'], marker=marker_lis[i], ms=5, ls=':', color=cmap(i), label = '%s' %tup[0])
        else:
            for i in range(len(mask_lis)):
                if mask_lis[i] in tup[0]:
                    ax1.plot(plt_dat[tup]['size_aero'], plt_dat[tup]['qual'], marker=marker_lis[i], ms=5, color=cmap(i), label = '%s' %tup[0])
    ax1.set_xscale('log')
    ax1.set_xticks(aero_lis)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xlabel('Aerodynamic Diameter [um]')
    ax1.set_ylabel('Quality Factor [1/kPa]')

    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_xaxis().set_tick_params(which='minor', size=0)
    ax1.get_xaxis().set_tick_params(which='minor', width=0)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('.\Data\Figures\Wash_Size\Wash_Size_Res_Aero_Qual.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    
# Location of summary data
filName = 'Penitration_Dat_08272021.csv'
filPath = r'.\Data\CSV'

# Read data into pandas data frame
df = pd.read_csv('%s\%s' %(filPath, filName))

# Convert to a dictonary containing numpy arrays
allDat = {}
bins = []
for header in df:
    if is_number(header):
        bins.append(header)
    allDat[header] = np.array(df[header])
bins = np.array(bins)

filter_lis = []
for key in allDat['Filter']:
    if (key not in filter_lis) and ('Mask' not in key):
        filter_lis.append(key)
layer_lis = ['SMT_FB_THCK', 'OLFN', 'ADVANCHK_SMS', 'H400', 'H100', 'PELL_930', 'FLANNEL', 'LEVEL 2 S.M.', 'SATEEN', 'FILTI', 'KONA ORNG'] #
mask_lis = ['Mask A', 'Mask B', 'Mask C', 'Mask D', 'Mask E', 'Mask F', 'Mask G', 'Mask A WASH', 'Mask B WASH', 'Mask C WASH', 'Mask D WASH', 'Mask E WASH', 'Mask F WASH', 'Mask G WASH']
mat_lis = ['SB', 'SB', 'SMS', 'SMS', 'SMS', 'SB', 'ML']



size_wash(allDat, [0.57], layer_lis) # 0.3, 0.37, 0.46, 0.71, 0.88, 1.09, 1.35, 1.68, 2.08, 2.58, 3.2


#qual_plot(allDat, 0.57, filter_lis)
#qual_plot_wash(allDat, 2.58, mask_lis)
quit()

# MODEL PLOTS
#layer_plt_masks_press(allDat, layer_lis, 'Y')

#layer_plt_masks(allDat, layer_lis, '0.57', 'N', [0, 15])

# 
#qual_plot_volt(allDat, 0.3, layer_lis, mat_lis)


# MODEL VS. THEORETICAL RESULTS
dat_dict = {'Mask A': [[74, 0.9856, 58], [79, 0.9859, 54]],
            'Mask B': [[95, 0.9953, 56], [90, 0.9855, 47]],
            'Mask C': [[60, 0.9986, 109], [60, 0.9921, 80]],
            'Mask D': [[34, 0.7540, 42], [79, 1, float('nan')]],
            'Mask E': [[74, 1, float('nan')], [71, 1, float('nan')]],
            'Mask F': [[44, 0.8251, 40], [44, 0.7763, 34]],
            'Mask G': [[67, 0.9135, 37], [91, 0.9200, 28]]}

cmap = get_cmap(len(dat_dict) + 1)
for i, mask in enumerate(dat_dict):
    plt.scatter(dat_dict[mask][0][0], dat_dict[mask][0][1], color=cmap(i), label='%s Theoretical' %(mask))
    plt.scatter(dat_dict[mask][1][0], dat_dict[mask][1][1], facecolors='none', edgecolors=cmap(i), label='%s Measured' %(mask))
    plt.plot([dat_dict[mask][0][0], dat_dict[mask][1][0]], [dat_dict[mask][0][1], dat_dict[mask][1][1]], color=cmap(i))

press_max = 100
eff_max = 1.01
x_rng = np.linspace(0, press_max, 256)
y_rng = np.ones_like(x_rng)
qual_lis = [0.5, 2, 5, 10, 20, 50, 100, 200, 500]
for Q in qual_lis:
        for i in range(len(y_rng)):
            y_rng[i] = qual_eff(Q, x_rng[i])
        plt.plot(x_rng, y_rng, linestyle='dotted', color='#858585', alpha=0.5)

plt.xlabel('Pressure Drop [Pa]')
plt.ylabel('Filtration Efficency')
plt.xlim([30, press_max])
plt.ylim([0.7, eff_max])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('.\Data\Figures\Qual\Qual_Plot_Mask_Stacks_4.2.svg', format='svg', bbox_inches='tight')
plt.show()
plt.close()

quit()

for material in layer_lis:
        #layer_plt(allDat, material, 'Y')
        layer_plt(allDat, material, 'N')
        #layer_plt_press(allDat, material, 'Y')
        layer_plt_press(allDat, material, 'N')

#
quit()


#layer_lis = ['H100', 'LEVEL 2 S.M.', 'SATEEN', 'KONA ORNG']
bin_lis = ['0.3', '0.37', '0.46', '0.57', '0.71', '0.88', '1.09', '1.35', '1.68', '2.08', '2.58', '3.2', '3.96']

#layer_plt_masks(allDat, layer_lis, '2.58', 'Y', [0, 15])
layer_plt_masks(allDat, layer_lis, '2.58', 'N', [0, 15])
#layer_plt_masks_press(allDat, layer_lis, 'Y')
layer_plt_masks_press(allDat, layer_lis, 'N')
quit()



#


#penitration_plt(allDat, bins, [61, 64])



idx_lis = [1]
for idx in idx_lis:
    pen_plt_overlay(allDat, 'Mask G', 'Mask G WASH', 'N', 'N', [idx])



layers = [1, 2, 3]
for layer in layers:
    pen_plt(allDat, 'H400', layer, 'Y')

