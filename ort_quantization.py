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

import onnx

from optimum.intel import OVQuantizer, OVConfig, OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSpeechSeq2Seq
from transformers import AutoTokenizer

import nncf
from nncf.onnx.quantization.backend_parameters import BackendParameters

ROOT = Path(__file__).parent.resolve()


MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"
# MODEL_ID = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
# MODEL_ID = "openai/tiny-whisper"
OUTPUT_DIR = ROOT / "tinyllama_compressed"


def main():
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    # model.save_pretrained(OUTPUT_DIR)

    # model = ORTModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny", export=True)
    # model.save_pretrained("./whisper_tiny")
    # exit(0)

    # onnx_model = onnx.load(OUTPUT_DIR / "model.onnx", load_external_data=False)
    # compressed_onnx_model = nncf.compress_weights(
    #     onnx_model,
    #     mode=nncf.CompressWeightsMode.INT8_ASYM,
    #     advanced_parameters=nncf.AdvancedCompressionParameters(
    #         backend_params={BackendParameters.EXTERNAL_DATA_DIR: OUTPUT_DIR}
    #     ),
    # )
    # onnx.save(compressed_onnx_model, OUTPUT_DIR / "model.onnx", save_as_external_data=True)

    OVQuantizer(model).quantize(
        save_directory=OUTPUT_DIR,
        ov_config=OVConfig(
            quantization_config=OVWeightQuantizationConfig(
                bits=4,
                dataset="wikitext2",
                tokenizer=MODEL_ID,
                ignored_scope=dict(types=["Gather"]),
                scale_estimation=True,
            ),
        )
    )

    # Infer Model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ov_model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR, from_onnx=True)
    ov_model = ORTModelForCausalLM.from_pretrained(OUTPUT_DIR)

    input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device=model.device)

    start_t = time.time()
    output = ov_model.generate(**input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
