import warnings

import matplotlib.font_manager
from matplotlib.pyplot import figure
import numpy as np
import seaborn as sns

from arch import arch_model
import arch.data.sp500

warnings.simplefilter("ignore")
sns.set_style("whitegrid")
sns.mpl.rcParams["figure.figsize"] = (12, 3)

data = arch.data.sp500.load()
market = data["Adj Close"]
returns = 100 * market.pct_change().dropna()

am = arch_model(returns)
res = am.fit(update_freq=5)

prop = matplotlib.font_manager.FontProperties("Roboto")


def _set_tight_x(axis, index):
    try:
        axis.set_xlim(index[0], index[-1])
    except ValueError:
        pass


fig = figure()
ax = fig.add_subplot(1, 1, 1)
vol = res.conditional_volatility
title = "S&P 500 Annualized Conditional Volatility"
scales = {"D": 252, "W": 52, "M": 12}
vol = vol * np.sqrt(scales["D"])

ax.plot(res._index.values, vol)
_set_tight_x(ax, res._index)
ax.set_title(title)
sns.despine(ax=ax)
title = ax.get_children()[7]
title.set_fontproperties(prop)
title.set_fontsize(26)
title.set_fontweight("ultralight")
title.set_fontstretch("ultra-condensed")
title.set_color("#757575")
ax.xaxis.label.set_color("#757575")
ax.yaxis.label.set_color("#757575")
ax.tick_params(axis="x", colors="#757575")
ax.tick_params(axis="y", colors="#757575")
fig.tight_layout(pad=1.0)
fig.savefig("hero.svg", transparent=True)
fig.savefig("hero.png", transparent=True)
