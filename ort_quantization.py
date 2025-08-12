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
from transformers import AutoProcessor, AutoTokenizer, set_seed

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


parser = argparse.ArgumentParser(description="Run model quantization and inference.")
parser.add_argument("--task", type=str, choices=["text-generation", "text-to-image", "automatic-speech-recognition"], help="Task to perform.")
parser.add_argument("--apply-quantization", action="store_true", help="Apply quantization to the model.")
parser.add_argument("--export-backend", type=str, default="onnx", choices=["onnx", "openvino"], help="Export backend to use.")
parser.add_argument("--inference-backend", type=str, default="openvino", choices=["onnx", "openvino"], help="Inference backend to use.")
parser.add_argument("--n-iter", type=int, default=1, help="Number of iterations for inference.")
args = parser.parse_args()


def main():
    if args.export_backend == "openvino" and args.inference_backend == "onnx":
        raise ValueError("Cannot run ORT inference on an openvino model")
    
    #
    # Prepare model and quantization configuration
    #
    if args.task == "text-generation":
        ort_model_cls, ov_model_cls = ORTModelForCausalLM, OVModelForCausalLM
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        quantization_config = OVWeightQuantizationConfig(
            bits=4,
            ignored_scope=dict(types=["Gather"]),
            scale_estimation=True,
            dataset="wikitext2",
            tokenizer=model_id,
        )
    elif args.task == "text-to-image":
        ort_model_cls, ov_model_cls = ORTDiffusionPipeline, OVDiffusionPipeline
        model_id = "stabilityai/stable-diffusion-2-1"
        quantization_config = OVWeightQuantizationConfig(
            bits=8,
            num_samples=200,
            dataset="conceptual_captions",
            quant_method=OVQuantizationMethod.HYBRID,
        )
    elif args.task == "automatic-speech-recognition":
        ort_model_cls, ov_model_cls = ORTModelForSpeechSeq2Seq, OVModelForSpeechSeq2Seq
        model_id = "openai/whisper-medium"
        quantization_config = OVQuantizationConfig(dataset="librispeech", processor=model_id, num_samples=32)
    else:
        raise ValueError(f"Unsupported args.task: {args.task}")

    precision_label = "quantized" if args.apply_quantization else "full-precision"
    output_dir = Path(".") / "ort_quantized_models" / model_id.split("/")[-1] / args.export_backend / precision_label
    output_dir = output_dir.absolute()
    export_cls = ov_model_cls if args.export_backend == "openvino" else ort_model_cls
    inference_cls = ov_model_cls if args.inference_backend == "openvino" else ort_model_cls

    #
    # Run the model export and optionally quantization
    #
    model_kwargs = {}
    if args.export_backend == "openvino":
        model_kwargs["load_in_8bit"] = False
    model = export_cls.from_pretrained(model_id, export=True, **model_kwargs)
    if args.apply_quantization:
        OVQuantizer(model).quantize(
            save_directory=output_dir, ov_config=OVConfig(quantization_config=quantization_config)
        )
    else:
        model.save_pretrained(output_dir)
    if args.task == "text-generation":
        AutoTokenizer.from_pretrained(model_id).save_pretrained(output_dir)
    elif args.task == "automatic-speech-recognition":
        AutoProcessor.from_pretrained(model_id).save_pretrained(output_dir)
    
    #
    # Run inference
    #
    if args.task == "text-generation":
        ov_model = inference_cls.from_pretrained(output_dir, from_onnx=args.export_backend == "onnx")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        messages = [{"role": "user", "content": "What is PyTorch?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        times = []
        for _ in range(args.n_iter):
            set_seed(0)
            start_t = time.time()
            output = ov_model.generate(input_ids, max_new_tokens=100, eos_token_id=-1)     # Fix number of output tokens
            times.append(time.time() - start_t)
        elapsed_time = f"Average elapsed time over {args.n_iter} iterations: {np.mean(times):.2f} seconds"
        print(elapsed_time)

        output_text = tokenizer.decode(output[0])
        with open(output_dir / f"output_{args.inference_backend}.txt", "w") as f:
            f.write(output_text + "\n" + elapsed_time)
        print(f'Model output: "{output_text.strip()}"')
    elif args.task == "automatic-speech-recognition":
        ov_model = inference_cls.from_pretrained(output_dir, from_onnx=args.export_backend == "onnx")
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
        times = []
        for _ in range(args.n_iter):
            set_seed(0)
            start_time = time.time()
            transcription = ov_model.generate(input_features)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        transcription = processor.batch_decode(transcription, skip_special_tokens=True)[0]
        print(f'\nQuantized model transcription: "{transcription.strip()}"')
        elapsed_time = f"Average elapsed time over {args.n_iter} iterations: {np.mean(times):.2f} seconds"
        print(elapsed_time)
        with open(output_dir / f"transcription_{args.inference_backend}.txt", "w") as f:
            f.write(transcription.strip() + "\n" + elapsed_time)
    elif args.task == "text-to-image":
        ov_model = inference_cls.from_pretrained(output_dir, from_onnx=args.export_backend == "onnx")

        # Generate an image.
        prompt = "A painting of a squirrel eating a burger"
        times = []
        for _ in range(args.n_iter):
            set_seed(0)
            start_t = time.time()
            images = ov_model(prompt, num_inference_steps=50, guidance_scale=7.5).images
            times.append(time.time() - start_t)
        elapsed_time = f"Average elapsed time over {args.n_iter} iterations: {np.mean(times):.2f} seconds"
        print(elapsed_time)
        with open(output_dir / f"generation_time_{args.inference_backend}.txt", "w") as f:
            f.write(elapsed_time)

        for i, image in enumerate(images):
            image.save(output_dir / f"generated_image_{args.inference_backend}_{i}.png")
        print(f"Generated {len(images)} images and saved to {output_dir}")
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
