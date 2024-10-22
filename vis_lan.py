import requests
from PIL import Image
from transformers import AutoProcessor

from optimum.intel.openvino import OVModelForVisualCausalLM, OVWeightQuantizationConfig
from optimum.intel.openvino.configuration import OVQuantizationMethod


data_aware_config = OVWeightQuantizationConfig(
    bits=4,
    sym=True,
    quant_method=OVQuantizationMethod.AWQ,
    num_samples=2,
    dataset="contextual",
    processor="llava-hf/llava-v1.6-mistral-7b-hf",
    group_size=16,
)

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = OVModelForVisualCausalLM.from_pretrained(model_id, load_in_8bit=False)
# model = OVModelForVisualCausalLM.from_pretrained(model_id, export=True, quantization_config=data_aware_config)
# model = OVModelForVisualCausalLM.from_pretrained(model_id, quantization_config=OVWeightQuantizationConfig(bits=4, sym=False))
# model.save_pretrained("llava/int4")

# model = OVModelForVisualCausalLM.from_pretrained("/home/nsavel/workspace/models/hf/llava-v1.6-mistral-7b-hf/FP32", compile=False)

# OVQuantizer(model).quantize(ov_config=OVConfig(quantization_config=data_aware_config))
# exit(0)
# model.save_pretrained("llava/int4_sym_awq_32")


# model = LlavaNextForConditionalGeneration.from_pretrained(model_id)
# patch_llava_vit(model)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))

print(model.vision_embeddings._infer_times)
print(model.language_model._infer_times)
