# Security Fix: Remote Code Execution in OVQuantizer

## Vulnerability Description

The OpenVINO quantization flow in `optimum.intel` accepted a remote `openvino_config.json` that could contain `trust_remote_code` and `tokenizer`/`processor` references. These values were forwarded into `AutoTokenizer.from_pretrained` and `AutoProcessor.from_pretrained` calls without sanitization, allowing remote code execution.

An attacker could publish a benign-looking model repository with a malicious `openvino_config.json` containing:
```json
{
    "quantization_config": {
        "tokenizer": "attacker-controlled-repo",
        "trust_remote_code": true
    }
}
```

When a victim loaded this config and ran quantization, the tool would import and execute code from the attacker's repository.

## Fix Implementation

### Changes Made

1. **Added `trust_remote_code` parameter to `OVQuantizer.quantize()` method**
   - Defaults to `False` for security
   - Overrides any `trust_remote_code` value in the quantization config
   - Documented in method docstring

2. **Propagated parameter through the call chain**
   - Updated `OVCalibrationDatasetBuilder.build_from_quantization_config()`
   - Updated `OVCalibrationDatasetBuilder.build_from_dataset_name()`
   - Updated `OVCalibrationDatasetBuilder.build_from_dataset()`
   - Updated all `_prepare_*_calibration_data()` methods

3. **Modified all AutoTokenizer/AutoProcessor calls**
   - All calls now use the user-provided `trust_remote_code` parameter
   - No longer use the potentially malicious `config.trust_remote_code` value

### Affected Methods

The following methods were updated to accept and use the `trust_remote_code` parameter:

- `_prepare_causal_lm_calibration_data()`
- `_prepare_visual_causal_lm_calibration_data()`
- `_prepare_speech_to_text_calibration_data()`
- `_prepare_text_to_text_calibration_data()`
- `_prepare_diffusion_calibration_data()`
- `_prepare_text_encoder_model_calibration_data()`
- `_prepare_text_image_encoder_model_calibration_data()`
- `_prepare_sam_dataset()`

## Usage Examples

### Safe Usage (Default)

```python
from optimum.intel import OVConfig, OVModelForCausalLM, OVQuantizer

model_id = "suspicious-model"
ov_config = OVConfig.from_pretrained(model_id, trust_remote_code=False)
model = OVModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)

# Safe: trust_remote_code defaults to False
quantizer = OVQuantizer(model)
quantizer.quantize(ov_config=ov_config, save_directory="./quantized_model")
```

### Explicit Override

```python
# Even if config has trust_remote_code=True, it will be overridden
quantizer = OVQuantizer(model)
quantizer.quantize(
    ov_config=ov_config, 
    save_directory="./quantized_model",
    trust_remote_code=False  # Explicitly override any config value
)
```

### Trusted Repository (Advanced Users)

```python
# Only for repositories you trust and have audited
quantizer = OVQuantizer(model)
quantizer.quantize(
    ov_config=ov_config, 
    save_directory="./quantized_model",
    trust_remote_code=True  # Use with caution!
)
```

## Security Recommendations

1. **Default to False**: The `trust_remote_code` parameter defaults to `False` for security
2. **Explicit Override**: Users must explicitly set `trust_remote_code=True` if they need custom code
3. **Validate Sources**: Only enable `trust_remote_code` for repositories you have audited
4. **Document Usage**: Always document why `trust_remote_code=True` is needed in your code

## Testing

The fix includes comprehensive unit tests in `tests/openvino/test_trust_remote_code_override.py` that verify:

1. The parameter properly overrides config values
2. AutoTokenizer.from_pretrained receives the correct trust_remote_code value
3. AutoProcessor.from_pretrained receives the correct trust_remote_code value
4. The override works across all preparation methods

## Backward Compatibility

This change is **backward compatible**:

- Existing code without the parameter will default to `trust_remote_code=False` (secure default)
- Code that was already passing `trust_remote_code` explicitly will continue to work
- The config's `trust_remote_code` value is now treated as untrusted and overridden

## Related Issues

- Original vulnerability report: [Issue Link]
- Security advisory: [Advisory Link]
