from . import (
    cosine_scheduler,
    parametric_peak_1,
    gaussian,
    lognormal,
    x_inverse,
    arctan
)

NODE_CLASS_MAPPINGS = {
    "CosineScheduler": cosine_scheduler.CosineScheduler,
    "Parametric Peak #1": parametric_peak_1.ParametricPeakNo1,
    "GaussianScheduler": gaussian.GaussianScheduler,
    "LogNormal Scheduler": lognormal.LogNormalScheduler,
    "k/x scheduler": x_inverse.X_InverseScheduler,
    "ArctanScheduler": arctan.Arctancheduler
}