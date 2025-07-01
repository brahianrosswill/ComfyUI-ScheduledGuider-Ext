from . import (
    scale_to_range,
    concat_sigmas,
    invert_sigmas,
    offset_sigmas,
    split_by_value,
    power,
    logarithm
)

NODE_CLASS_MAPPINGS = {
    "ScaleToRange": scale_to_range.StaleToRange,
    "InvertSigmas": invert_sigmas.InvertSigmas,
    "ConcatSigmas": concat_sigmas.ConcatSigmas,
    "OffsetSigmas": offset_sigmas.OffsetSigmas,
    "SplitSigmasByValue": split_by_value.SplitSigmasByValue,
    "SigmasToPower": power.SigmasToPower,
    "PredefinedExponent": power.PredefinedExponent,
    "CustomExponent": power.CustomExponent,
    "PredefinedLogarithm": logarithm.PredefinedLogarithm,
    "CustomBaseLogarithm": logarithm.CustomBaseLogarithm,
}