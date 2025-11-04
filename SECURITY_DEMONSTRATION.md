# Demonstration: Security Fix Effectiveness

This document demonstrates how the security fix prevents remote code execution attacks.

## Attack Scenario (Before Fix)

### Malicious Config File
An attacker creates a repository with the following `openvino_config.json`:

```json
{
    "quantization_config": {
        "tokenizer": "attacker-controlled-repo/malicious-tokenizer",
        "trust_remote_code": true
    }
}
```

### Victim Code (Before Fix)
```python
from optimum.intel import OVConfig, OVModelForCausalLM, OVQuantizer

# Victim loads the malicious config
model_id = "attacker-repo/benign-looking-model"
ov_config = OVConfig.from_pretrained(model_id, trust_remote_code=False)
model = OVModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)

# Victim runs quantization
quantizer = OVQuantizer(model)
quantizer.quantize(ov_config=ov_config, save_directory="./quantized_model")
```

### What Happened (Before Fix)
1. Victim loads config with `trust_remote_code=False` (safe)
2. Config contains `trust_remote_code=true` (malicious)
3. During quantization, the code calls:
   ```python
   AutoTokenizer.from_pretrained(
       "attacker-controlled-repo/malicious-tokenizer",
       trust_remote_code=True  # ← Used from config!
   )
   ```
4. **RESULT**: Attacker's code executes on victim's machine

## Protection (After Fix)

### Same Malicious Config
The attacker still has the same malicious config:

```json
{
    "quantization_config": {
        "tokenizer": "attacker-controlled-repo/malicious-tokenizer",
        "trust_remote_code": true
    }
}
```

### Victim Code (After Fix)
```python
from optimum.intel import OVConfig, OVModelForCausalLM, OVQuantizer

# Victim loads the malicious config
model_id = "attacker-repo/benign-looking-model"
ov_config = OVConfig.from_pretrained(model_id, trust_remote_code=False)
model = OVModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)

# Victim runs quantization (trust_remote_code defaults to False)
quantizer = OVQuantizer(model)
quantizer.quantize(ov_config=ov_config, save_directory="./quantized_model")
```

### What Happens (After Fix)
1. Victim loads config with `trust_remote_code=False` (safe)
2. Config contains `trust_remote_code=true` (malicious)
3. During quantization, the code now calls:
   ```python
   AutoTokenizer.from_pretrained(
       "attacker-controlled-repo/malicious-tokenizer",
       trust_remote_code=False  # ← OVERRIDDEN by parameter!
   )
   ```
4. **RESULT**: Attack is blocked! Custom code is not executed.

## Code Flow Comparison

### Before Fix
```
User calls quantize()
  → reads config.trust_remote_code (attacker controlled)
    → passes to AutoTokenizer.from_pretrained()
      → EXECUTES MALICIOUS CODE
```

### After Fix
```
User calls quantize(trust_remote_code=False)  [default]
  → ignores config.trust_remote_code
    → uses parameter value (False)
      → passes to AutoTokenizer.from_pretrained()
        → BLOCKS MALICIOUS CODE
```

## Explicit Override Example

If a user really needs to trust remote code (for legitimate reasons):

```python
# User explicitly enables trust_remote_code for a trusted repository
quantizer = OVQuantizer(model)
quantizer.quantize(
    ov_config=ov_config,
    save_directory="./quantized_model",
    trust_remote_code=True  # User explicitly allows this
)
```

This requires an explicit, conscious decision by the user, not a hidden value in a config file.

## Key Security Improvements

1. **Default Deny**: `trust_remote_code` defaults to `False`
2. **User Control**: Config values cannot override user's security choices
3. **Explicit Intent**: Users must explicitly enable remote code execution
4. **Audit Trail**: It's clear in the code where `trust_remote_code=True` is used

## Verification

The fix can be verified by checking:

1. All `AutoTokenizer.from_pretrained()` calls use `trust_remote_code` parameter
2. All `AutoProcessor.from_pretrained()` calls use `trust_remote_code` parameter
3. The parameter is passed through the entire call chain
4. Unit tests verify the override behavior

## Lines Changed

Total changes in `optimum/intel/openvino/quantization.py`:
- 1 method signature updated with new parameter
- 9 internal methods updated to propagate parameter
- 8 AutoTokenizer/AutoProcessor calls updated to use parameter
- Documentation added explaining the security parameter

This minimal, surgical change eliminates the vulnerability while maintaining backward compatibility.
