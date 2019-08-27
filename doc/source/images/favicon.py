import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 10)

start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2020, 1, 1)
data = pdr.get_data_yahoo('^GSPC', start, end)
price = data['Adj Close']
rets = 100 * price.resample('M').last().pct_change()

l, u = rets.quantile([.01, .99])
bins = np.linspace(l, u, 11)
fig = plt.figure(frameon=False)
fig.set_size_inches(8, 8)
ax = fig.add_subplot('111')
rwidth = np.diff(bins).mean() * 0.333
ax.hist(rets, bins=bins, rwidth=rwidth, align='mid', color='#4caf50ff')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('')
sns.despine(left=True, bottom=True)
fig.tight_layout(pad=1.0)

fig.savefig('favicon.svg', transparent=True, bbox_inches=0)
fig.savefig('favicon.png', transparent=True)

fig = plt.figure(frameon=False)
fig.set_size_inches(8, 8)
ax = fig.add_subplot('111')
rwidth = np.diff(bins).mean() * 0.333
ax.hist(rets, bins=bins, rwidth=rwidth, align='mid', color='#ffffff')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('')
sns.despine(left=True, bottom=True)
fig.tight_layout(pad=1.0)

fig.savefig('logo.svg', transparent=True, bbox_inches=0)
