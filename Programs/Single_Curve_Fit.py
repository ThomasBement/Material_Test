import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# Model for multilayer stacks
def layer_model(layers, a):
    return 1 - (np.exp(-a*layers)) #a*(b + (layers**c))

def layer_model_press(layers, a):
    return a*layers

x_rng = [0, 1.0, 2.0, 3.0]
y_rng = [0, 13.333333333333334, 27.333333333333332, 44.666666666666664]
y_err = [0.0001, 0.4714045207910317, 1.247219128924647, 1.247219128924647]

# Pen
"""
H100 [0, 1.0, 2.0, 3.0] [0, 0.7874929025, 0.95786747406, 1.0] [0.0001, 0.016664426150765785, 0.009218441593649129, 0.0]
LEVEL 2 S.M. [0, 1.0] [0, 1.0] [0.0001, 0.0]
SATEEN [0, 1.0] [0, 0.42961712739999997] [0.0001, 0.04073796689802008]
KONA ORNG [0, 1.0] [0, 0.3041277345] [0.0001, 0.12068044436483123]
"""

# Press
"""
H100 [0, 1.0, 2.0, 3.0] [0, 13.333333333333334, 27.333333333333332, 44.666666666666664] [0.0001, 0.4714045207910317, 1.247219128924647, 1.247219128924647]
LEVEL 2 S.M. [0, 1.0] [0, 44.0] [0.0001, 5.887840577551898]
SATEEN [0, 1.0] [0, 23.333333333333332] [0.0001, 1.8856180831641267]
KONA ORNG [0, 1.0] [0, 15.666666666666666] [0.0001, 0.9428090415820634]
"""

def fit_model(x_rng, y_rng, y_err, model):
    p0 = [1]

    popt, pcov = curve_fit(model, x_rng, y_rng, p0=p0, sigma=y_err, absolute_sigma=True)

    x_fit = np.linspace(min(x_rng), max(x_rng), 64)
    y_fit = model(x_fit, *popt)
    a = popt[0]

    plt.errorbar(x_rng, y_rng, yerr=y_err, marker='o', ls='none', capsize=3, elinewidth=1, markeredgewidth=2)
    plt.plot(x_fit, y_fit)
    plt.show()

    print(a)

fit_model(x_rng, y_rng, y_err, layer_model_press)