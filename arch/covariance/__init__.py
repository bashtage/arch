from typing import Dict, Type

from . import kernel

KERNEL_ESTIMATORS: Dict[str, Type[kernel.CovarianceEstimator]] = {
    est_name.lower(): getattr(kernel, est_name) for est_name in kernel.KERNELS
}
KERNEL_ESTIMATORS.update(
    {est_name: getattr(kernel, est_name) for est_name in kernel.KERNELS}
)
KNOWN_KERNELS = "\n".join(sorted([k for k in KERNEL_ESTIMATORS]))
KERNEL_ERR = f"kernel is not a known estimator. Must be one of:\n {KNOWN_KERNELS}"
