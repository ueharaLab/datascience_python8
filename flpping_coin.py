import numpy as np
from binom import coin_trials
import matplotlib.pyplot as plt

flips = 100
cf = coin_trials(flips)

while True:
    trials = int(input('input number of trials:'))
    #freq = cf.fit(trials)
    cf.fit(trials)
    plt.bar(np.arange(flips), cf.freq)    
    plt.show()