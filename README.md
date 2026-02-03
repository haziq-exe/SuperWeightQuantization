# Super-Weight Aware Quantization

Implementation of super-weight aware quantization from ["The Super Weight in Large Language Models"](https://arxiv.org/abs/2411.07191) (Yu et al., 2024, Apple). 

Super weights are 1-6 scalar values in an LLM that are critical for model quality, pruning them destroys the model's ability to generate coherent output. This implements INT4 weight quantization while preserving these parameters in FP16.

---

## Overview

The paper discovered that not all the "super weights" that models contain have the same influence on model output, in fact, they found that in most models removing only around 1-6 scalar values can completely degrade performance.

Yet removing the top 7000 other super weights barely affects performance. This implementation quantizes model weights to INT4 while preserving those top 1-6 weights in FP16. (further details of implementation mentioned in original paper)

---

## Features

- **INT4 Quantization:** Reduces model size with minimal quality loss
- **Super-Weight Preservation:** Maintains critical super weight values in FP16
- **Larger Block Sizes:** Uses 512×512 block quantization
- **OLMo Support:** Implements for OLMo-1B model only (extendible to other models but would need to change things like exact super weight co-ordinates)

---

## Requirements

```bash
pip install torch ai2-olmo
```

---

## How It Works

1. **Clip outliers** using z-score threshold (3.0, selected mostly arbritarily, paper didn't disclose z-score they used)
2. **Quantize/dequantize** to INT4
3. **Restore super weights** in FP16

For OLMo-1B, super weights are at Layer 1, `ff_out` weights `[1764, 1710]` and `[1764, 8041]`.

---

## Sources

- Paper: https://arxiv.org/abs/2411.07191
- OLMo Models: https://allenai.org/olmo

---
