from SuperWeightQuantizer import SuperWeightQuantizer

quantizer = SuperWeightQuantizer("allenai/OLMo-1B")
quantizer.load_model()

test_prompt = "Summer is hot. Winter is"
print("\n" + "="*60)
print("BEFORE QUANTIZATION")
print("="*60)
print(f"Prompt: {test_prompt}")
print(f"Generated: {quantizer.generate_text(test_prompt)}")

# Apply super-weight aware quantization with larger block size
results = quantizer.quantize_model(block_size=(512, 512))

print("\n" + "="*60)
print("AFTER QUANTIZATION")
print("="*60)
print(f"Prompt: {test_prompt}")
print(f"Generated: {quantizer.generate_text(test_prompt)}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for result in results:
    print(f"Layer {result['layer']} - {result['module']}")
    print(f"  Super weights preserved: {len(result['super_weights'])}")
    print(f"  Quantized blocks: {result['quantized_blocks']}")
    print(f"  Block size: {result['block_size']}")