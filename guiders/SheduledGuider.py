import numpy

import comfy.samplers
from comfy_extras.nodes_perpneg import perp_neg


def find_clothest_index(sigma, sigma_triggers):
    index = 0
    for i, trigger_sigma in enumerate(sigma_triggers):
        if trigger_sigma >= sigma:
            index = i
        else:
            break
    return index


class Guider_SheduledCFG: # Removed inheritance from comfy.samplers.CFGGuider
    def __init__(self, model, cfg_max, cfg_min, sigmas, neg_scale=0, use_negative_as_unconditional=True):
        self.inner_model = model
        self.cfg_max = cfg_max
        self.cfg_min = cfg_min
        self.sigmas = sigmas # This should be the sigmas from the sampler/scheduler
        self.neg_scale = neg_scale
        self.use_negative_as_unconditional = use_negative_as_unconditional

        if sigmas is not None and len(sigmas) > 0:
            self.sigma_max = float(max(sigmas))
            self.sigma_min = float(min(sigmas))
            # model_sigma_triggers was calculated here but is unused.
            # The CFG scheduling correctly uses self.sigmas (KSampler's sigmas).
            # Removing model_sigma_triggers to simplify and avoid confusion.
        else:
            # Handle cases where sigmas might be None or empty
            self.sigma_max = 0.0
            self.sigma_min = 0.0
            print("Warning: Sigmas not provided or empty during Guider_SheduledCFG initialization.")
            # self.model_sigma_triggers would have been an empty numpy array here.

    def get_model_object(self, key):
        return self.inner_model.get_model_object(key)

    def set_use_negative(self, use_neg: bool):
        self.use_negative_as_unconditional = use_neg

    # set_conds, get_conditions, calc_predictions, calc_cfg are removed / integrated.

    def predict_noise(self, x, sigma, cond, uncond, cond_scale, model_options={}, negative_cond=None):
        current_sigma_val = sigma.item() if hasattr(sigma, 'item') else sigma

        # Use self.sigmas (from KSampler) to find the current position in the schedule
        closest_index_in_ksampler_sigmas = find_clothest_index(current_sigma_val, self.sigmas)
        sigma_for_cfg_calc = self.sigmas[closest_index_in_ksampler_sigmas]

        if self.sigma_max > self.sigma_min:
            current_percent = (sigma_for_cfg_calc - self.sigma_min) / (self.sigma_max - self.sigma_min)
        else:
            current_percent = 1.0
        
        # Ensure current_percent is clamped between 0 and 1 before interpolation
        current_percent = max(0.0, min(1.0, current_percent))
        current_cfg = self.cfg_min + (self.cfg_max - self.cfg_min) * current_percent
        
        # Get predictions from the wrapped model.
        # cond_scale=1.0 because we are handling scaling externally.
        uncond_pred = self.inner_model.predict_noise(x, sigma, uncond, cond_scale=1.0, model_options=model_options)
        cond_pred = self.inner_model.predict_noise(x, sigma, cond, cond_scale=1.0, model_options=model_options)

        if negative_cond is not None and self.neg_scale > 0:
            neg_pred = self.inner_model.predict_noise(x, sigma, negative_cond, cond_scale=1.0, model_options=model_options)
            
            cfg_result = perp_neg(
                x,            # Not used by perp_neg, but part of its signature
                cond_pred,    # Positive prediction
                neg_pred,     # Negative prediction
                uncond_pred,  # Unconditional prediction
                self.neg_scale, # Guidance scale for negative prompt
                current_cfg   # Master guidance scale (scheduled CFG)
            )

            for fn in model_options.get("sampler_post_cfg_function", []):
                args = {
                    "denoised": cfg_result,
                    "cond": cond,
                    "uncond": negative_cond if self.use_negative_as_unconditional else uncond,
                    "model": self.inner_model, # The original wrapped model
                    "uncond_denoised": neg_pred if self.use_negative_as_unconditional else uncond_pred,
                    "cond_denoised": cond_pred,
                    "sigma": sigma,
                    "model_options": model_options,
                    "input": x,
                }
                # Add empty_cond and empty_cond_denoised if use_negative_as_unconditional is true,
                # mimicking the structure from the original calc_cfg.
                if self.use_negative_as_unconditional:
                    args["empty_cond"] = uncond # The true unconditional
                    args["empty_cond_denoised"] = uncond_pred # Denoised for true unconditional
                
                cfg_result = fn(args)
            return cfg_result
        else:
            # Standard CFG: uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            cfg_result = uncond_pred + current_cfg * (cond_pred - uncond_pred)

            for fn in model_options.get("sampler_post_cfg_function", []):
                args = {
                    "denoised": cfg_result,
                    "cond": cond,
                    "uncond": uncond,
                    "model": self.inner_model, # The original wrapped model
                    "uncond_denoised": uncond_pred,
                    "cond_denoised": cond_pred,
                    "sigma": sigma,
                    "model_options": model_options,
                    "input": x,
                }
                cfg_result = fn(args)
            return cfg_result


class SheduledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "unconditional": ("CONDITIONING", ),
                "cfg_max": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "cfg_min": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "sigmas": ("SIGMAS", ),
                }}

    RETURN_TYPES = ("MODEL",) # CRITICAL FIX: Was "GUIDER"

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self, model, positive, unconditional, cfg_max, cfg_min, sigmas
    ):
        # The SheduledCFGGuider node now returns the wrapper directly.
        # Conditioning will be handled by the sampler via predict_noise args.
        # ksampler will take this wrapped_model and use it.
        # The positive and unconditional inputs to this node are effectively ignored
        # as the ksampler will provide its own conditioning.
        # However, to maintain the node signature and avoid breaking existing workflows
        # that might connect them, we accept them but don't use them here.
        
        wrapped_model = Guider_SheduledCFG(model, cfg_max, cfg_min, sigmas)
        # We return the wrapped_model as a "MODEL" type, because it now duck-types
        # a model with a predict_noise method.
        return (wrapped_model,)


class PerpNegSheduledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "unconditional": ("CONDITIONING", ),
                "cfg_max": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "cfg_min": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "neg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                }),
                "sigmas": ("SIGMAS", ),
                "use_negative_as_unconditional": ("BOOLEAN", {
                    "default": True
                }),
                }}

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive, # Ignored, ksampler provides
            negative, # Ignored, ksampler provides via negative_cond to predict_noise
            unconditional, # Ignored, ksampler provides
            cfg_max,
            cfg_min,
            neg_scale,
            sigmas,
            use_negative_as_unconditional
    ):
        # Similar to SheduledCFGGuider, this node now returns the wrapper.
        # Conditioning is handled by the sampler.
        wrapped_model = Guider_SheduledCFG(
            model,
            cfg_max,
            cfg_min,
            sigmas,
            neg_scale,
            use_negative_as_unconditional # Passed to __init__
        )
        # The set_use_negative method is still useful if one wanted to change it post-init,
        # but here it's part of construction.
        # If use_negative_as_unconditional was not part of __init__, then:
        # wrapped_model.set_use_negative(use_negative_as_unconditional)
        
        # Return the wrapped_model as a "MODEL" type.
        return (wrapped_model,)
