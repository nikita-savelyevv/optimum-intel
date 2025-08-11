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

import argparse
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from optimum.intel import (
    OVConfig,
    OVDiffusionPipeline,
    OVModelForSpeechSeq2Seq,
    OVQuantizationConfig,
    OVQuantizer,
    OVWeightQuantizationConfig,
)
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.configuration import OVQuantizationMethod
from optimum.onnxruntime import ORTDiffusionPipeline, ORTModelForCausalLM, ORTModelForSpeechSeq2Seq


# TASK = "text-generation"
# TASK = "text-to-image"
# TASK = "automatic-speech-recognition"
# APPLY_QUANTIZATION = bool(1)
# USE_OV_FOR_INFERENCE = bool(1)

parser = argparse.ArgumentParser(description="Run model quantization and inference.")
parser.add_argument(
    "--task",
    type=str,
    choices=["text-generation", "text-to-image", "automatic-speech-recognition"],
    help="Task to perform.",
)
parser.add_argument("--apply-quantization", action="store_true", help="Apply quantization to the model.")
parser.add_argument(
    "--use-ov-for-inference", action="store_true", help="Use OpenVINO for inference instead of ONNX Runtime."
)
args = parser.parse_args()
TASK = args.task
APPLY_QUANTIZATION = args.apply_quantization
USE_OV_FOR_INFERENCE = args.use_ov_for_inference


def main():
    if TASK == "text-generation":
        ort_model_cls, ov_model_cls = ORTModelForCausalLM, OVModelForCausalLM
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        quantization_config = OVWeightQuantizationConfig(
            # bits=8,
            # sym=True,
            bits=4,
            # all_layers=True,
            ignored_scope=dict(types=["Gather"]),
            scale_estimation=True,
            dataset="wikitext2",
            tokenizer=model_id,
        )
    elif TASK == "text-to-image":
        ort_model_cls, ov_model_cls = ORTDiffusionPipeline, OVDiffusionPipeline
        model_id = "stabilityai/stable-diffusion-2-1"
        quantization_config = OVWeightQuantizationConfig(
            bits=8,
            num_samples=200,
            dataset="conceptual_captions",
            quant_method=OVQuantizationMethod.HYBRID,
        )
    elif TASK == "automatic-speech-recognition":
        ort_model_cls, ov_model_cls = ORTModelForSpeechSeq2Seq, OVModelForSpeechSeq2Seq
        # model_id = "openai/whisper-tiny"
        model_id = "openai/whisper-medium"
        quantization_config = OVQuantizationConfig(dataset="librispeech", processor=model_id, num_samples=32)
    else:
        raise ValueError(f"Unsupported TASK: {TASK}")

    model_label = "quantized" if APPLY_QUANTIZATION else "full-precision"
    output_dir = (Path(".") / "ort_quantized_models" / model_id.split("/")[-1] / model_label).absolute()
    inference_cls = ov_model_cls if USE_OV_FOR_INFERENCE else ort_model_cls

    # Run the model export and optionally quantization
    model = ort_model_cls.from_pretrained(model_id, export=True)
    if APPLY_QUANTIZATION:
        OVQuantizer(model).quantize(
            save_directory=output_dir, ov_config=OVConfig(quantization_config=quantization_config)
        )
    else:
        model.save_pretrained(output_dir)

    # Run inference
    if ort_model_cls == ORTModelForCausalLM:
        ov_model = inference_cls.from_pretrained(output_dir, from_onnx=USE_OV_FOR_INFERENCE)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        messages = [{"role": "user", "content": "What is PyTorch?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        start_t = time.time()
        output = ov_model.generate(input_ids, max_new_tokens=100)
        elapsed_time = f"Elapsed time: {time.time() - start_t:.2f} seconds"
        print(elapsed_time)

        output_text = tokenizer.decode(output[0])
        with open(output_dir / "output.txt", "w") as f:
            f.write(output_text + "\n" + elapsed_time)
        print(f'Model output: "{output_text.strip()}"')
    elif ort_model_cls == ORTModelForSpeechSeq2Seq:
        ov_model = inference_cls.from_pretrained(output_dir, from_onnx=USE_OV_FOR_INFERENCE)
        processor = AutoProcessor.from_pretrained(model_id)

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
        elapsed_time = f"Elapsed time: {time.time() - start_time:.2f} seconds"
        print(f'\nQuantized model transcription: "{transcription.strip()}"')
        print(elapsed_time)
        with open(output_dir / "transcription.txt", "w") as f:
            f.write(transcription.strip() + "\n" + elapsed_time)
    elif ort_model_cls == ORTDiffusionPipeline:
        ov_model = inference_cls.from_pretrained(output_dir, from_onnx=USE_OV_FOR_INFERENCE)

        # Generate an image.
        prompt = "A painting of a squirrel eating a burger"
        start_t = time.time()
        images = ov_model(prompt, num_inference_steps=50, guidance_scale=7.5).images
        elapsed_time = f"Elapsed time: {time.time() - start_t:.2f} seconds"
        print(elapsed_time)
        with open(output_dir / "generation_time.txt", "w") as f:
            f.write(elapsed_time)

        for i, image in enumerate(images):
            image.save(output_dir / f"generated_image_{i}.png")
        print(f"Generated {len(images)} images and saved to {output_dir}")
    else:
        raise ValueError(f"Unsupported model class: {ort_model_cls}")


if __name__ == "__main__":
    main()
