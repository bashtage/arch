import datetime

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
import seaborn as sns

colors = sns.color_palette("muted")

NBINS = 7

plt.rcParams["figure.figsize"] = (10, 10)

start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2020, 1, 1)
data = pdr.get_data_yahoo("^GSPC", start, end)
price = data["Adj Close"]
rets = 100 * price.resample("M").last().pct_change()

lower, upper = rets.quantile([0.01, 0.99])
bins = np.linspace(lower, upper, NBINS)
fig = plt.figure(frameon=False)
fig.set_size_inches(8, 8)
ax = fig.add_subplot("111")
rwidth = np.diff(bins).mean() * 0.22
_, _, patches = ax.hist(rets, bins=bins, rwidth=rwidth, align="mid")  # '#2196f3')
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i])
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel("")
sns.despine(left=True, bottom=True)
fig.tight_layout(pad=1.0)

fig.savefig("favicon.svg", transparent=True, bbox_inches=0)
fig.savefig("favicon.png", transparent=True)

fig = plt.figure(frameon=False)
fig.set_size_inches(8, 8)
ax = fig.add_subplot("111")
rwidth = np.diff(bins).mean() * 0.22
ax.hist(rets, bins=bins, rwidth=rwidth, align="mid", color="#ffffff")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel("")
sns.despine(left=True, bottom=True)
fig.tight_layout(pad=1.0)

fig.savefig("logo.svg", transparent=True, bbox_inches=0)


prop = matplotlib.font_manager.FontProperties("Roboto")
prop.set_size(216)
fig = plt.figure(frameon=False, figsize=(12, 4))
ax = fig.add_subplot("111")
rwidth = np.diff(bins).mean() * 0.22
_, _, patches = ax.hist(rets, bins=bins, rwidth=rwidth, align="mid")
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i])
ax.set_xticks([])
ax.set_yticks([])
text = ax.text(11, 0, "arch")
text.set_fontproperties(prop)
text.set_fontsize(216)
text.set_color("#757575")
ax.set_ylim(0, 180)
sns.despine(left=True, bottom=True)
fig.tight_layout(pad=1.0)
fig.savefig("color-logo.png", transparent=True)
fig.savefig("color-logo.svg", transparent=True)
