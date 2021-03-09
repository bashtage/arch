import sys
from typing import Optional
import warnings


def check_reindex(reindex: Optional[bool]) -> bool:
    default = False if "arch.__future__.reindexing" in sys.modules else True

    if reindex is None:
        if default:
            warnings.warn(
                """
The default for reindex is True. After September 2021 this will change to
False. Set reindex to True or False to silence this message. Alternatively,
you can use the import comment

from arch.__future__ import reindexing

to globally set reindex to True and silence this warning.
""",
                FutureWarning,
            )
        reindex = default

    return reindex
