import os

import numpy as np

FILE_TYPES = ("z", "p")
TRENDS = ("n", "c", "ct", "ctt")

if os.name == "posix":
    ROOT = "/mnt/c/Users/kevin/Dropbox/phillips-ouliaris"
else:
    ROOT = r"c:\Users\kevin\Dropbox\phillips-ouliaris"

# percentiles to save
PERCENTILES = (
    list(np.arange(1, 10, 1))
    + list(np.arange(10, 990, 5))
    + list(np.arange(990, 1000, 1))
)
PERCENTILES = PERCENTILES[::-1]
QUANTILES = np.array(PERCENTILES) / 1000.0

SAMPLE_SIZES = (
    25,
    30,
    35,
    40,
    45,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    160,
    180,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
    1200,
    1400,
    2000,
)
