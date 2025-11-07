# ruff: noqa
import hashlib
import pickle
import tempfile
import time
from pathlib import Path

import nncf
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, set_seed

from optimum.intel import OVModelForTextToSpeechSeq2Seq, OVWeightQuantizationConfig, OVPipelineQuantizationConfig

# SAMPLING_RATE = 16000
SAMPLING_RATE = 22050
SAVE_DIR = Path("tts")


def compute_mcd(reference, target):
    """Compute Mel-Cepstral Distortion (MCD) between reference and synthesized audio."""
    from pymcd.mcd import Calculate_MCD

    mcd_tool = Calculate_MCD(MCD_mode="dtw")
    score = mcd_tool.calculate_mcd(reference, target)
    return score


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)


def load_model(backend, model_id, vocoder_id, quantization_config=None):
    if backend == "ov":
        set_seed(0)
        model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            vocoder=vocoder_id,
            load_in_8bit=False,
            quantization_config=quantization_config,
        )
    else:
        model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    processor = SpeechT5Processor.from_pretrained(model_id)
    return model, processor


def infer_model(model, processor, text, speaker_embeddings=None):
    inputs = processor(text=text, return_tensors="pt")

    if isinstance(model, OVModelForTextToSpeechSeq2Seq):
        start_time = time.perf_counter()
        speech = model.generate(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        infer_time = time.perf_counter() - start_time
        speech = speech.numpy()[0]
    else:
        set_seed(0)
        start_time = time.perf_counter()
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan"),
        )
        infer_time = time.perf_counter() - start_time
        speech = speech.numpy()

    speech = resample(speech, src_sample_rate=16000, dst_sample_rate=SAMPLING_RATE)

    return speech, infer_time


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def demo_predictions(model, processor, text, save_filename, speaker_embeddings=None):
    """
    Generate audio predictions for a given text input using the specified model and processor.
    """
    speech, _ = infer_model(model, processor, text, speaker_embeddings)
    sf.write(SAVE_DIR / save_filename, speech, samplerate=SAMPLING_RATE)
    print(f"Audio saved to {SAVE_DIR / save_filename}. MD5: {compute_md5(SAVE_DIR / save_filename)}")
    return speech


def get_dataset(dataset_size=100, max_n_words=None):
    dataset_file = Path("tts/dataset.txt")
    if dataset_file.exists():
        with open(dataset_file, "r") as f:
            samples = eval(f.read())
        assert len(samples) >= dataset_size, "Dataset file has fewer samples than requested."
        return samples[:dataset_size]
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
    dataset = dataset.shuffle(seed=42).take(dataset_size)
    samples = []
    for data_item in tqdm(dataset, total=dataset_size, desc="Downloading dataset"):
        text = data_item["text"]
        if max_n_words is not None:
            text = " ".join(text.strip().split(maxsplit=max_n_words + 1)[:max_n_words])
        samples.append(text)
    # with open("tts_dataset.txt", "w") as f:
    #     f.write(str(samples))
    return samples


def collect_speech_references(model, processor, speaker_embeddings, dataset):
    references = []
    times = []
    durations = []
    for text in tqdm(dataset, desc="Computing speech references"):
        speech, infer_time = infer_model(model, processor, text, speaker_embeddings)
        times.append(infer_time)
        durations.append(len(speech) / SAMPLING_RATE)
        references.append(speech)
    avg_time = sum(times) / sum(durations)
    return references, avg_time


def compute_mcd_targets(target_model, processor, speaker_embeddings, dataset, references, save_dir=None):
    mcd_scores = []
    times = []
    durations = []
    avg_durations = []
    for i, text in tqdm(enumerate(dataset), total=len(dataset), desc="Computing MCD"):
        ref_speech = references[i]
        target_speech, infer_time = infer_model(target_model, processor, text, speaker_embeddings)
        times.append(infer_time)
        durations.append(len(target_speech) / SAMPLING_RATE)
        avg_durations.append((len(target_speech) + len(ref_speech)) / (2 * SAMPLING_RATE))

        clean_up = False
        if save_dir is None:
            clean_up = True
            save_dir = tempfile.mkdtemp()
        save_dir.mkdir(parents=True, exist_ok=True)
        ref_path = Path(save_dir) / f"ref_{i}.wav"
        target_path = Path(save_dir) / f"target_{i}.wav"
        sf.write(ref_path, ref_speech, SAMPLING_RATE)
        sf.write(target_path, target_speech, SAMPLING_RATE)
        mcd = compute_mcd(ref_path.as_posix(), target_path.as_posix())
        mcd_scores.append(mcd)
        if clean_up:
            ref_path.unlink()
            target_path.unlink()

    avg_time = sum(times) / sum(durations)

    # Average MCD scores
    mcd_scores = np.array(mcd_scores, dtype=np.float32)
    avg_durations = np.array(avg_durations, dtype=np.float32)
    linear_mcd_scores = np.power(10, mcd_scores / 10)
    weighted_linear_mcd_scores = np.sum(linear_mcd_scores * avg_durations) / np.sum(avg_durations)
    average_mcd = 10 * np.log10(weighted_linear_mcd_scores)
    return average_mcd, avg_time


def examine_low_precision_constants(model_path):
    from openvino.runtime import Core

    core = Core()
    ov_model = core.read_model(model=model_path)
    for op in ov_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_element_type().get_type_name() == "u8":
            name = op.get_friendly_name()
            if "zero_point" not in name:
                continue
            data = op.data.astype(np.float32)
            # Print tabulated row for mean, median, min, max, std
            print(
                f"mean: {data.mean():.4f} | median: {np.median(data):.4f} | "
                f"min: {data.min():.4f} | max: {data.max():.4f} | std: {data.std():.4f} | Name: {name}"
            )


if __name__ == "__main__":
    # examine_low_precision_constants("/home/nsavel/workspace/optimum-intel/tts/int8/openvino_decoder_model.xml")
    # exit(0)

    model_id = "microsoft/speecht5_tts"
    vocoder_id = "microsoft/speecht5_hifigan"
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


    ref_model, processor = load_model("pt", model_id, vocoder_id)
    fp32_label = "fp32"
    target_model_fp32, _ = load_model("ov", model_id, vocoder_id)
    target_model_fp32.save_pretrained(SAVE_DIR / fp32_label)

    # int8_label = "int8"
    # int8_label = "int8_wo-decoder"
    # int8_label = "int8_decoder-ignored-scope"
    int8_label = "int8_ptq"
    # quantization_config = OVWeightQuantizationConfig(bits=8)
    quantization_config = OVPipelineQuantizationConfig(
        quantization_configs={
            "encoder": OVWeightQuantizationConfig(bits=8),
            # "decoder": OVWeightQuantizationConfig(bits=8,),
            "decoder": OVWeightQuantizationConfig(
                bits=8,
                ignored_scope={"patterns": [
                    "__module.speech_decoder_postnet",
                    "__module.speecht5.decoder.prenet",
                ]}
            ),
            "vocoder": OVWeightQuantizationConfig(bits=8),
            "postnet": OVWeightQuantizationConfig(bits=8),
        }
    )
    target_model_int8, _ = load_model("ov", model_id, vocoder_id, quantization_config)

    # hacky PTQ
    with open("tts/decoder_inputs.pkl" , "rb") as f:
        decoder_inputs = pickle.load(f)
    target_model_int8.decoder.model = nncf.quantize(
        target_model_fp32.decoder.model,
        calibration_dataset=nncf.Dataset([decoder_inputs]),
        ignored_scope=nncf.IgnoredScope(patterns=["__module.speech_decoder_postnet", "__module.speecht5.decoder.prenet"])
    )

    target_model_int8.save_pretrained(SAVE_DIR / int8_label)

    text = "Hello, this pull request introduces support of SpeechT5 text-to-speech pipeline using OpenVINO."
    demo_predictions(ref_model, processor, text, "demo_speech_pt.wav", speaker_embeddings)
    demo_predictions(target_model_fp32, processor, text, f"{fp32_label}/demo_speech.wav", speaker_embeddings)
    demo_predictions(target_model_int8, processor, text, f"{int8_label}/demo_speech.wav", speaker_embeddings)
    # exit(0)

    dataset_size, max_n_words = 100, 50
    test_dataset = get_dataset(dataset_size=dataset_size, max_n_words=max_n_words)
    references, time_ref = collect_speech_references(ref_model, processor, speaker_embeddings, test_dataset)
    print(f"Reference model avg. inference time: {time_ref:.4f} sec.")
    # for model, model_label in [(target_model_fp32, fp32_label)]:
    for model, model_label in [(target_model_fp32, fp32_label), (target_model_int8, int8_label)]:
        mcd, time_target = compute_mcd_targets(
            model, processor, speaker_embeddings, test_dataset, references, SAVE_DIR / model_label / "mcd_files"
        )
        print(f"{model_label} model avg. inference time: {time_target:.4f} s")
        print(f"MCD {model_label}: {mcd:.4f}")
        with open(SAVE_DIR / model_label.lower() / "mcd.txt", "w") as f:
            # Write time and MCD for each model
            f.write(f"Average inference time: {time_target:.4f} sec.\n")
            f.write(f"MCD: {mcd:.4f} (dataset size: {dataset_size}, max n words: {max_n_words})\n")
