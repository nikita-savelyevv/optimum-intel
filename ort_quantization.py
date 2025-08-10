# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pathlib import Path
import numpy as np
import onnx

from optimum.intel import OVQuantizer, OVConfig, OVWeightQuantizationConfig, OVQuantizationConfig, OVDiffusionPipeline, \
    OVModelForSpeechSeq2Seq
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSpeechSeq2Seq, ORTDiffusionPipeline
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset

import nncf
from nncf.onnx.quantization.backend_parameters import BackendParameters

from optimum.intel.openvino.configuration import OVQuantizationMethod

ROOT = Path(__file__).parent.resolve()


# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_ID = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
MODEL_ID = "openai/whisper-tiny"
# MODEL_ID = "stabilityai/stable-diffusion-2-1"
# OUTPUT_DIR = ROOT / "ort_quantized_model/tiny-llama-se"
# OUTPUT_DIR = ROOT / "ort_quantized_model/stable-diffusion-2-1/hq"
OUTPUT_DIR = ROOT / "ort_quantized_model/tmp"


def main():
    # ort_model_cls, ov_model_cls = ORTModelForCausalLM, OVModelForCausalLM
    ort_model_cls, ov_model_cls = ORTModelForSpeechSeq2Seq, OVModelForSpeechSeq2Seq
    # ort_model_cls, = ORTDiffusionPipeline, OVDiffusionPipeline

    model = ort_model_cls.from_pretrained(MODEL_ID, export=True)
    # model.save_pretrained(OUTPUT_DIR)
    OVQuantizer(model).quantize(
        save_directory=OUTPUT_DIR,
        ov_config=OVConfig(
            quantization_config=OVQuantizationConfig(dataset="librispeech", processor=MODEL_ID)
    #         # quantization_config=OVWeightQuantizationConfig(
    #         #     bits=8,
    #         #     sym=True,
    #         #     # bits=4,
    #         #     # all_layers=True,
    #         #     # ignored_scope=dict(types=["Gather"]),
    #         #     # scale_estimation=True,
    #         #     # dataset="wikitext2",
    #         #     # tokenizer=MODEL_ID,
    #         # ),
    #         # quantization_config=OVWeightQuantizationConfig(
    #         #     bits=8,
    #         #     num_samples=200,
    #         #     dataset="conceptual_captions",
    #         #     quant_method=OVQuantizationMethod.HYBRID,
    #         # )
        )
    )

    # Infer Model.
    if ort_model_cls == ORTModelForCausalLM:
        # ov_model = ov_model_cls.from_pretrained(OUTPUT_DIR, from_onnx=True)
        ov_model = ort_model_cls.from_pretrained(OUTPUT_DIR)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        messages = [{"role": "user", "content": "What is PyTorch?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        start_t = time.time()
        output = ov_model.generate(input_ids, max_new_tokens=100)
        print("Elapsed time: ", time.time() - start_t)

        output_text = tokenizer.decode(output[0])
        print(output_text)
        return output_text
    elif ort_model_cls == ORTModelForSpeechSeq2Seq:
        ov_model = ov_model_cls.from_pretrained(OUTPUT_DIR, from_onnx=True)
        # ov_model = ort_model_cls.from_pretrained(OUTPUT_DIR)
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        def extract_input_features(sample):
            audio = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            if sampling_rate != 16000:
                duration = audio.shape[0] / sampling_rate
                resampled_data = np.zeros(shape=(int(duration * 16000)), dtype=np.float32)
                x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
                x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
                audio = np.interp(x_new, x_old, audio)

            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            return input_features

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        input_features = extract_input_features(dataset[0])
        start_time = time.time()
        transcription = processor.batch_decode(ov_model.generate(input_features), skip_special_tokens=True)[0]
        print("Elapsed time: ", time.time() - start_time)
        print(f'\nQuantized model transcription: "{transcription.strip()}"')
    elif ort_model_cls == ORTDiffusionPipeline:
        # ov_model = ov_model_cls.from_pretrained(OUTPUT_DIR, from_onnx=True)
        ov_model = ort_model_cls.from_pretrained(OUTPUT_DIR)

        # Generate an image.
        prompt = "A painting of a squirrel eating a burger"
        start_t = time.time()
        images = ov_model(prompt, num_inference_steps=50, guidance_scale=7.5).images
        print("Elapsed time: ", time.time() - start_t)

        for i, image in enumerate(images):
            image.save(OUTPUT_DIR / f"generated_image_{i}.png")
        print(f"Generated {len(images)} images and saved to {OUTPUT_DIR}")
    else:
        raise ValueError(f"Unsupported model class: {ort_model_cls}")


if __name__ == "__main__":
    main()
