class StaleToRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "sigma_min": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.1,
                    "min": -5000,
                    "max": 5000,
                }),
                "sigma_max": ("FLOAT", {
                    "default": 8.0,
                    "step": 0.1,
                    "min": -5000,
                    "max": 5000,
                }),
            },
        }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, sigma_min, sigma_max):
        min_value = min(sigmas)
        max_value = max(sigmas)
        steps = len(sigmas)
        minmax_delta = sigma_max - sigma_min
        value_minmax_delta = max_value - min_value
        scale_coef = minmax_delta / value_minmax_delta
        for step in range(steps):
            sigmas[step] = (sigmas[step] - min_value) * scale_coef + sigma_min
        return (sigmas,)