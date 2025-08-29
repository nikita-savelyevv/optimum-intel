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
from typing import Optional

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from jiwer import wer, wer_standardize
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm
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


def extract_input_features(processor, sample):
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


parser = argparse.ArgumentParser(description="Run model quantization and inference.")
parser.add_argument(
    "--task",
    type=str,
    choices=["text-generation", "text-to-image", "automatic-speech-recognition"],
    help="Task to perform.",
)
parser.add_argument("--apply-quantization", action="store_true", help="Apply quantization to the model.")
parser.add_argument(
    "--export-backend", type=str, default="onnx", choices=["onnx", "openvino"], help="Export backend to use."
)
parser.add_argument(
    "--inference-backend", type=str, default="openvino", choices=["onnx", "openvino"], help="Inference backend to use."
)
parser.add_argument("--n-iter", type=int, default=1, help="Number of iterations for inference.")
parser.add_argument("--validate", action="store_true", help="Validate instead of exporting")
args = parser.parse_args()


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


def main():
    #
    # Run the model export and optionally quantization
    #
    model_kwargs = {}
    if args.export_backend == "openvino":
        model_kwargs["load_in_8bit"] = False
        if args.task == "automatic-speech-recognition":
            model_kwargs["stateful"] = False
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
            output = ov_model.generate(input_ids, max_new_tokens=100, eos_token_id=-1)  # Fix number of output tokens
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

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        input_features = extract_input_features(processor, dataset[0])
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
            images = ov_model(prompt).images
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


def validate_whisper(test_dataset_size):
    processor = AutoProcessor.from_pretrained(model_id)

    def calculate_transcription_time_and_accuracy(ov_model, test_samples):
        infer_times = []

        ground_truths = []
        predictions = []
        for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
            input_features = extract_input_features(processor, data_item)

            start_time = time.time()
            predicted_ids = ov_model.generate(input_features)
            infer_times.append(time.time() - start_time)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            ground_truths.append(data_item["text"])
            predictions.append(transcription[0])
            # print(f"Ground truth: {data_item['text']}")
            # print(f"Prediction: {transcription[0]}")

        word_accuracy = (
            1
            - wer(
                ground_truths, predictions, reference_transform=wer_standardize, hypothesis_transform=wer_standardize
            )
        ) * 100
        mean_infer_time = sum(infer_times) / len(infer_times)
        return word_accuracy, mean_infer_time

    test_dataset = load_dataset(
        "openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True
    )
    test_dataset = test_dataset.shuffle(seed=0).take(test_dataset_size)
    test_samples = [sample for sample in test_dataset]

    model = inference_cls.from_pretrained(output_dir, from_onnx=args.export_backend == "onnx")

    accuracy, mean_infer_time = calculate_transcription_time_and_accuracy(model, test_samples)
    print(f"Word accuracy: {accuracy:.2f}%")
    print(f"Average inference time: {mean_infer_time:.4f} seconds")
    with open(output_dir / "validation.txt", "w") as f:
        f.write(f"Word accuracy: {accuracy:.2f}%\n")
        f.write(f"Average inference time: {mean_infer_time:.4f} seconds\n")


def validate_text_to_image(
    test_dataset_size: int,
    batch_size: int = 64,
    real_images_dir: Optional[str] = None,    # folder with *real* images for FID (optional)
    clip_model_name_or_path: str = "openai/clip-vit-base-patch16",  # per torchmetrics docs
):
    """
    Evaluates a text-to-image model with:
      - CLIPScore (torchmetrics) — semantic text↔image alignment (higher is better)
      - (optional) FID (torchmetrics) vs. a directory of real images (lower is better)

    Notes:
      - Requires globals present in your env: `inference_cls`, `output_dir`, `args` (with .export_backend)
      - Saves generated images into <output_dir>/validation_images
    """

    def _image_paths_in_dir(root: str):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        root = Path(root)
        return [str(p) for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Dataset with prompts -----
    dataset = load_dataset(
        "google-research-datasets/conceptual_captions",
        "unlabeled",
        split="validation",
        trust_remote_code=True,
    ).shuffle(seed=42)
    dataset = dataset.take(test_dataset_size)

    # ----- Your model under test -----
    model = inference_cls.from_pretrained(output_dir, from_onnx=args.export_backend == "onnx")

    # ----- Output paths -----
    images_save_dir = Path(output_dir) / "validation_images"
    images_save_dir.mkdir(parents=True, exist_ok=True)

    # ----- Generation loop -----
    prompts = []
    gen_tensors = []   # store as CHW float in [0,1] for metrics
    infer_times = []

    to_tensor = T.ToTensor()  # PIL -> float tensor in [0,1]

    print("Generating images...")
    for row in tqdm(dataset, desc="Generation"):
        prompt = row["caption"]

        # Skip overly long prompts if your tokenizer has limits
        if hasattr(model, "tokenizer"):
            max_len = getattr(model.tokenizer, "model_max_length", None)
            if max_len is not None and len(prompt) > max_len:
                continue

        start = time.perf_counter()
        pil_img = model(prompt).images[0]  # PIL.Image
        infer_times.append(time.perf_counter() - start)

        # Save image
        safe_name = prompt[:80].replace(" ", "_").replace("/", "_")
        out_path = images_save_dir / f"{safe_name}.png"
        pil_img.save(out_path)

        # Keep for metrics
        prompts.append(prompt)
        gen_tensors.append(to_tensor(pil_img))  # CHW, float32 in [0,1]

    if len(gen_tensors) == 0:
        raise RuntimeError("No images were generated (all prompts filtered out?).")

    mean_perf_time = sum(infer_times) / len(infer_times)

    # ----- CLIPScore (torchmetrics) -----
    # This downloads/loads the CLIP model internally and computes the *average* score.
    clip_metric = CLIPScore(model_name_or_path=clip_model_name_or_path).to(device)
    clip_metric.reset()

    print("Computing CLIPScore (torchmetrics)...")
    for i in range(0, len(gen_tensors), batch_size):
        imgs = torch.stack(gen_tensors[i : i + batch_size]).to(device)   # [B,3,H,W], float in [0,1]
        txts = prompts[i : i + batch_size]                               # List[str]
        clip_metric.update(imgs, txts)

    clip_score = float(clip_metric.compute().item())  # scalar average (higher is better)

    # ----- (Optional) FID (torchmetrics) -----
    fid_score = None
    if real_images_dir is not None:
        real_paths = _image_paths_in_dir(real_images_dir)
        if len(real_paths) == 0:
            print(f"[WARN] No images found in real_images_dir: {real_images_dir}. Skipping FID.")
        else:
            print(f"Computing FID vs. {real_images_dir} (torchmetrics, size-agnostic)...")
            # normalize=True -> inputs in [0,1] float
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            fid_metric.reset()

            to_tensor = T.ToTensor()  # PIL -> float in [0,1], shape [3,H,W]

            # --- Update REAL distribution, one-by-one (any HxW works) ---
            for rp in tqdm(real_paths, desc="FID: real (streaming)"):
                try:
                    real_img = Image.open(rp).convert("RGB")
                except Exception:
                    continue
                rb = to_tensor(real_img).unsqueeze(0).to(device)  # [1,3,H,W]
                fid_metric.update(rb, real=True)

            # --- Update FAKE (generated) distribution, also one-by-one to be size-agnostic ---
            for gt in tqdm(gen_tensors, desc="FID: fake (streaming)"):
                gb = gt.unsqueeze(0).to(device)  # [1,3,H,W]
                fid_metric.update(gb, real=False)

            fid_score = float(fid_metric.compute().item())

    # ----- Report -----
    print(f"CLIPScore (avg): {clip_score:.4f}")
    if fid_score is not None:
        print(f"FID: {fid_score:.2f}  (lower is better)")
    else:
        print("FID: skipped (no real_images_dir provided)")
    print(f"Average inference time: {mean_perf_time:.4f} seconds")

    # ----- Persist to file -----
    with open(Path(output_dir) / "validation.txt", "w", encoding="utf-8") as f:
        f.write(f"CLIPScore (avg): {clip_score:.4f}\n")
        if fid_score is not None:
            f.write(f"FID: {fid_score:.2f}\n")
        else:
            f.write("FID: skipped (no real_images_dir provided)\n")
        f.write(f"Average inference time: {mean_perf_time:.4f} seconds\n")


if __name__ == "__main__":
    if args.validate:
        if args.task == "automatic-speech-recognition":
            validate_whisper(test_dataset_size=args.n_iter)
        elif args.task == "text-to-image":
            validate_text_to_image(
                test_dataset_size=args.n_iter,
                real_images_dir="/media/hdd1/datasets/coco/images/val2017"
            )
        else:
            raise ValueError(
                "Validation is only supported for the automatic-speech-recognition and text-to-image tasks."
            )
    else:
        main()
