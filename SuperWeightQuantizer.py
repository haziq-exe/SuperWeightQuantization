"""
Super-Weight Aware Quantization Implementation
Based on "The Super Weight in Large Language Models" (Yu et al., 2024) (https://arxiv.org/abs/2411.07191)

Implements weight quantization with super-weight preservation for OLMo-1B.
Super weights are critical scalar values that disproportionately impact model quality.
"""

import torch
import torch.nn as nn
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import numpy as np
from typing import Tuple, List, Dict


class SuperWeightQuantizer:
    """
    Quantizes LLM weights while preserving super weights.
    
    Super weights are 1-6 scalar values per model that are critical for quality.
    Pruning them completely destroys the model's ability to generate text.
    
    Process (from paper Section 4.2):
    1. Clip outlier weights using z-score threshold
    2. Quantize and dequantize to INT4
    3. Restore super weights in FP16
    """
    
    def __init__(self, model_name = "allenai/OLMo-1B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Super weight coordinates from Table 2 of paper
        self.super_weights = {
            "allenai/OLMo-1B": [
                (1, "mlp.down_proj", 1764, 1710),
                (1, "mlp.down_proj", 1764, 8041)
            ]
        }
        
    def load_model(self, device):

        print(f"Loading {self.model_name}...")
        self.model = OLMoForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto"
        ).to(device)
        self.tokenizer = OLMoTokenizerFast.from_pretrained(self.model_name)
    
    def clip_weights(self, W, z_threshold = 3.0):
        """Paper doesn't mention exact z-score they used, from what I know 3.0 is common. (unsure though)"""
        mean = W.mean()
        std = W.std()
        
        z_scores = (W - mean) / std
        
        W_clipped = W.clone()
        outlier_mask = torch.abs(z_scores) > z_threshold
        
        upper_bound = mean + z_threshold * std
        lower_bound = mean - z_threshold * std
        
        W_clipped[outlier_mask] = torch.clamp(
            W_clipped[outlier_mask], 
            lower_bound, 
            upper_bound
        )
        
        return W_clipped
    
    def quantize_weights_int4(self, W, block_size = (128, 128)):
        """
        INT4 asymmetric round-to-nearest quantization.
        
        Q(X) = Round((X - MIN) / delta)
        delta = (MAX - MIN) / (2^N - 1)
        """
        D, H = W.shape
        block_rows, block_cols = block_size
        
        N = 4  # 4-bit quantization
        n_levels = 2 ** N
        
        quantized_blocks = []
        quant_params = []
        
        for i in range(0, D, block_rows):
            for j in range(0, H, block_cols):
                block = W[i:min(i+block_rows, D), j:min(j+block_cols, H)]
                
                min_val = block.min()
                max_val = block.max()
                delta = (max_val - min_val) / (n_levels - 1)
                
                quantized = torch.round((block - min_val) / delta)
                quantized = torch.clamp(quantized, 0, n_levels - 1).to(torch.int8)
                
                quantized_blocks.append(quantized)
                quant_params.append({
                    'min': min_val,
                    'max': max_val,
                    'delta': delta,
                    'position': (i, j)
                })
        
        return quantized_blocks, quant_params
    
    def dequantize_weights(self, quantized_blocks, quant_params, original_shape, device = None):
        """
        Dequantize INT4 back to FP16.
        Q^-1(X_hat) = delta * X_hat + MIN
        """
        D, H = original_shape
        
        if device is None:
            device = quantized_blocks[0].device
        
        W_dequant = torch.zeros(D, H, dtype=torch.float16, device=device)
        
        for q_block, params in zip(quantized_blocks, quant_params):
            i, j = params['position']
            delta = params['delta']
            min_val = params['min']
            
            if not isinstance(delta, torch.Tensor):
                delta = torch.tensor(delta, dtype=torch.float16, device=device)
            if not isinstance(min_val, torch.Tensor):
                min_val = torch.tensor(min_val, dtype=torch.float16, device=device)
            
            q_block = q_block.to(device)
            
            block_dequant = delta * q_block.float() + min_val
            
            block_rows, block_cols = q_block.shape
            W_dequant[i:i+block_rows, j:j+block_cols] = block_dequant.to(torch.float16)
        
        return W_dequant
    
    def quantize_layer_with_super_weights(self, layer_idx, module_name, block_size = (512, 512), z_threshold = 3.0):
        """
        Apply super-weight aware quantization to layer.
        
        Equation (2) from paper:
        W_hat = RESTORE(Q^-1(Q(CLIP_z(W))))
        """
        layer = self.model.model.transformer.blocks[layer_idx]
        
        # in OlMo library mlp.down_proj corresponds to ff_out
        if 'down_proj' in module_name:
            module = layer.ff_out
            actual_module_name = 'ff_out'
        else:
            module = getattr(layer, module_name)
            actual_module_name = module_name
        
        W_original = module.weight.data.clone()
        original_device = W_original.device
        
        print(f"\nProcessing {actual_module_name} with shape {W_original.shape} on device {original_device}")
        
        # Find super weights
        super_weight_coords = []
        super_weight_values = []
        
        for sw in self.super_weights[self.model_name]:
            sw_layer, sw_module, row, col = sw
            if sw_layer == layer_idx and 'down_proj' in sw_module:
                if row < W_original.shape[0] and col < W_original.shape[1]:
                    super_weight_coords.append((row, col))
                    super_weight_values.append(W_original[row, col].item())
                    print(f"Found super weight at layer {layer_idx}, "
                          f"{actual_module_name}[{row}, {col}] = {W_original[row, col].item():.4f}")
                else:
                    print(f"WARNING: Super weight coordinate [{row}, {col}] out of bounds for shape {W_original.shape}")
        
        if not super_weight_coords:
            print(f"No super weights found for layer {layer_idx}")
            return None
        
        # Clip outliers
        W_clipped = self.clip_weights(W_original, z_threshold)
        
        print(f"Clipping stats - Original range: [{W_original.min():.4f}, {W_original.max():.4f}]")
        print(f"Clipping stats - Clipped range: [{W_clipped.min():.4f}, {W_clipped.max():.4f}]")
        
        # Quantize and dequantize
        print(f"Quantizing with block size {block_size}...")
        quantized_blocks, quant_params = self.quantize_weights_int4(W_clipped, block_size)
        W_dequant = self.dequantize_weights(
            quantized_blocks, 
            quant_params, 
            W_original.shape, 
            device=original_device
        )
        
        # Restore super weights
        W_final = W_dequant.clone()
        for (row, col), original_value in zip(super_weight_coords, super_weight_values):
            W_final[row, col] = original_value
            print(f"Restored super weight at [{row}, {col}] = {original_value:.4f}")
        
        W_final = W_final.to(original_device)
        module.weight.data = W_final
        
        return {
            'layer': layer_idx,
            'module': actual_module_name,
            'super_weights': super_weight_coords,
            'quantized_blocks': len(quantized_blocks),
            'block_size': block_size
        }
    
    def quantize_model(self, block_size = (512, 512)):
        
        if self.model is None:
            self.load_model()
        
        results = []
        
        print("\n" + "="*60)
        print("Starting Super-Weight Aware Quantization")
        print("="*60 + "\n")
        
        layers_to_quantize = set()
        for sw in self.super_weights[self.model_name]:
            layers_to_quantize.add((sw[0], sw[1]))
        
        for layer_idx, module_name in layers_to_quantize:
            print(f"\nProcessing Layer {layer_idx} - {module_name}")
            print("-" * 60)
            
            result = self.quantize_layer_with_super_weights(
                layer_idx, 
                module_name, 
                block_size=block_size
            )
            
            if result is not None:
                results.append(result)
        
        print("\n" + "="*60)
        print("Quantization Complete!")
        print("="*60)
        
        return results
    
    def evaluate_perplexity(self, text):

        if self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def generate_text(self, prompt, max_new_tokens = 50):
        
        if self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        message = [prompt]
        inputs = self.tokenizer(message, return_tensors='pt', return_token_type_ids=False)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=self.model.device)
                ], dim=-1)
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]