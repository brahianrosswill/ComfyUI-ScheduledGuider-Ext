import math

import numpy
import torch


class ParametricPeakNo1:
    """
    Parametric Peak Sigma Scheduler

    This node generates a sigma array with a customizable peak shape,
    useful for controlling the sampling process in diffusion models.
    The shape of the curve can be adjusted using 'peak', 'warmup', and 'decay'
    parameters.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 10000,
                    "display": "The total number of sampling steps."
                }),
                "peak": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.11,
                    "max": 0.91,
                    "step": 0.01,
                    "round": False,
                    "display": (
                        "The position of the peak in the sigma curve "
                        "(0.0 to 1.0, relative to total steps)."
                    )
                }),
                "warmup": ("FLOAT", {
                    "default": 1.0,
                    "min": -0.99,
                    "max": 1.95,
                    "step": 0.01,
                    "round": False,
                    "display": (
                        "Controls the shape of the curve before the peak. "
                        "Positive values create a concave shape, "
                        "negative values a convex shape, "
                        "and 0 creates a linear ramp."
                    )
                }),
                "decay": ("FLOAT", {
                    "default": 1.0,
                    "min": -0.99,
                    "max": 1.95,
                    "step": 0.01,
                    "round": False,
                    "display": (
                        "Controls the shape of the curve after the peak. "
                        "Positive values create a concave shape, "
                        "negative values a convex shape, "
                        "and 0 creates a linear decay."
                    )
                }),
            }
        }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/CFG-schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps: int, peak: float, warmup: float, decay: float):
        sigmas = numpy.zeros(steps)
        peak_step = int(round(steps * peak))
        warmup_period = math.pi * min(warmup, 1)
        warmup_delay = int(round(peak_step * (max(abs(warmup), 1) - 1)))
        warmup_delta_s = 1 / (peak_step - warmup_delay) * warmup_period
        if warmup == 0:
            warmup_delta_s = 1 / (peak_step - warmup_delay)
        warmup_min = math.cos(warmup_period)
        warmup_divider = 1 - warmup_min
        warmup_offset = \
            (warmup_min * -1) / warmup_divider if warmup != 0 else 0
        decay_steps = steps - peak_step
        decay_period = min(decay, 1) * math.pi
        decay_release = int(round(decay_steps * (max(abs(decay), 1) - 1)))
        decay_release_step = peak_step + decay_steps - decay_release
        decay_delta_s = 1 / (decay_steps - decay_release) * decay_period
        if decay == 0:
            decay_delta_s = 1 / (decay_steps - decay_release)
        decay_min = math.cos(decay_period)
        decay_divider = 1 - decay_min
        decay_offset = (decay_min * -1) / decay_divider if decay != 0 else 0

        for step in range(steps):
            if step < peak_step:
                if step < warmup_delay:
                    sigmas[step] = 0
                else:
                    x = (peak_step - step) * warmup_delta_s
                    if warmup < 0:
                        x = (step - warmup_delay) * warmup_delta_s
                        sigmas[step] = \
                            1 - (math.cos(x) / warmup_divider + warmup_offset)
                    elif warmup > 0:
                        sigmas[step] = \
                            math.cos(x) / warmup_divider + warmup_offset
                    else:
                        sigmas[step] = 1 - x
            else:
                if step < decay_release_step:
                    x = (step - peak_step) * decay_delta_s
                    if decay > 0:
                        sigmas[step] = \
                            math.cos(x) / decay_divider + decay_offset
                    elif decay < 0:
                        x = (steps - step) * decay_delta_s
                        sigmas[step] = \
                            1 - (math.cos(x) / decay_divider + decay_offset)
                    else:
                        sigmas[step] = 1 - x
                else:
                    sigmas[step] = 0

        return (torch.FloatTensor(sigmas), )
