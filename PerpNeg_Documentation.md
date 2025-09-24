# Documentation for Perpendicular Negative Guidance (Perp-Neg) Nodes

This document provides a detailed explanation of the Perpendicular Negative Guidance (Perp-Neg) nodes found in `nodes_perpneg.py`.

## Core Concept: Perpendicular Negative Guidance

Perpendicular Negative Guidance is an advanced technique for guiding diffusion models during image generation. It refines the standard Classifier-Free Guidance (CFG) by preventing the negative prompt from directly removing elements from the positive prompt. Instead, it steers the generation process away from the negative prompt in a direction that is "perpendicular" to the positive prompt's guidance.

### How it Works

In standard CFG, the guidance is a simple subtraction: `final_guidance = positive_guidance - negative_guidance`. This can sometimes lead to the negative prompt "canceling out" related concepts in the positive prompt, resulting in a loss of desired detail or a washed-out image.

Perp-Neg modifies this by calculating a "perpendicular" component of the negative guidance. It projects the negative guidance vector onto the positive guidance vector and then subtracts this projection from the original negative guidance. The result is a guidance vector that is orthogonal (perpendicular) to the positive guidance.

`perp_neg_guidance = negative_guidance - projection(negative_guidance, positive_guidance)`

This `perp_neg_guidance` is then scaled and subtracted from the positive guidance. The effect is that the model is steered away from the negative prompt's concepts without directly conflicting with the positive prompt's concepts.

## `PerpNegGuider` Node

This is the main node for applying Perpendicular Negative Guidance. It takes the place of the standard KSampler's guidance system.

### Input Parameters

*   **`model`**: The diffusion model to be guided.
*   **`positive`**: The positive conditioning (prompt). This defines what you *want* to see in the image.
*   **`negative`**: The negative conditioning (prompt). This defines what you *do not* want to see in the image.
*   **`empty_conditioning`**: This is the unconditional conditioning, usually an empty prompt. It represents the model's "unbiased" prediction and is crucial for both standard CFG and Perp-Neg.
*   **`cfg` (float)**: This is the standard CFG scale. It controls the overall strength of the positive prompt's influence.
    *   **Practical Effect**: Higher `cfg` values make the image adhere more strictly to the positive prompt. A `cfg` of 0 would ignore the prompt entirely, while very high values can lead to oversaturated and distorted images. A typical range is 7-10.
*   **`neg_scale` (float)**: This is the "Perpendicular Negative Scale". It controls the strength of the perpendicular negative guidance.
    *   **Practical Effect**: This parameter is unique to Perp-Neg. It determines how strongly the model should avoid the concepts in the negative prompt, but in a way that doesn't conflict with the positive prompt.
        *   `neg_scale = 0.0`: Perpendicular guidance is turned off. The sampler behaves like standard CFG.
        *   `neg_scale = 1.0`: This is the default and standard strength for perpendicular guidance. It's a good starting point.
        *   `neg_scale > 1.0`: Increases the effect, pushing the image further away from the negative prompt's concepts. This can be useful for combating persistent unwanted elements. For example, if you are generating a forest scene and want to avoid any buildings, a higher `neg_scale` can be more effective than just putting "buildings" in the negative prompt with standard CFG.
        *   `neg_scale < 1.0`: Decreases the effect.

### Example Use Case

Imagine you want a picture of a "beautiful fantasy landscape." You might also add a negative prompt of "photorealistic, ugly, plain."

*   With **standard CFG**, the "photorealistic" in the negative prompt might inadvertently reduce the "fantasy" elements in the positive prompt, making the image less stylized.
*   With **Perp-Neg**, the `neg_scale` would guide the image away from photorealism without directly subtracting from the "fantasy" concept. This can result in a more vibrant and imaginative image that still respects the negative prompt.

## `PerpNeg` Node (Deprecated)

The `nodes_perpneg.py` file also contains a `PerpNeg` node. This is an older, deprecated implementation that patches the model directly. It is less efficient and flexible than the `PerpNegGuider` node.

**It is strongly recommended to use the `PerpNegGuider` node instead.** The `PerpNeg` node is kept for backward compatibility but should not be used in new workflows.