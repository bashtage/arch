import arch.unitroot as u

import numpy as np

#y = np.random.randn(5299788)
#adf = u.ADF(y, trend='ct')
#adf.stat
#import time

#print('sleeping')
#time.sleep(5)
#print('back')
y = np.random.randn(5299788)
adf = u.ADF(y, trend='ct', low_memory=True)
print(adf.stat)
print(adf.lags)
