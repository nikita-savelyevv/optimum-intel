#  Copyright 2024 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Integration tests for dataset option parsing with calibration data preparation."""

import unittest

from optimum.intel.openvino.configuration import OVQuantizationConfigBase


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset kwargs threading through calibration helpers."""

    def test_causal_lm_seq_len_from_dataset_kwargs(self):
        """Test that seq_len from dataset_kwargs is used in causal LM calibration."""
        config = OVQuantizationConfigBase(dataset="wikitext2:seq_len=256", tokenizer="gpt2", num_samples=2)

        # Verify parsing
        self.assertEqual(config.dataset, "wikitext2")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 256})

    def test_gsm8k_custom_seq_len_overrides_default(self):
        """Test that custom seq_len for gsm8k overrides the default 256."""
        config = OVQuantizationConfigBase(dataset="gsm8k:seq_len=512", tokenizer="gpt2", num_samples=2)

        # Verify parsing
        self.assertEqual(config.dataset, "gsm8k")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 512})

    def test_text_to_text_seq_len_from_kwargs(self):
        """Test that seq_len can be passed via dataset_kwargs to text-to-text helper."""
        config = OVQuantizationConfigBase(dataset="c4:seq_len=256", tokenizer="t5-small", num_samples=2)

        # Verify parsing
        self.assertEqual(config.dataset, "c4")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 256})

        # The helper should receive seq_len=256 as a keyword argument
        # when unpacking **config.dataset_kwargs

    def test_text_encoder_seq_len_from_kwargs(self):
        """Test that seq_len can be passed via dataset_kwargs to text encoder helper."""
        config = OVQuantizationConfigBase(dataset="wikitext:seq_len=64", tokenizer="bert-base-uncased", num_samples=2)

        # Verify parsing
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 64})

    def test_backward_compatibility_no_options(self):
        """Test that datasets without options work as before."""
        configs = [
            OVQuantizationConfigBase(dataset="wikitext2", tokenizer="gpt2"),
            OVQuantizationConfigBase(dataset="gsm8k", tokenizer="gpt2"),
            OVQuantizationConfigBase(dataset="c4", tokenizer="t5-small"),
        ]

        for config in configs:
            self.assertEqual(config.dataset_kwargs, {})

    def test_list_dataset_backward_compatibility(self):
        """Test that list datasets work unchanged."""
        dataset_list = ["This is text 1", "This is text 2"]
        config = OVQuantizationConfigBase(dataset=dataset_list, tokenizer="gpt2")

        self.assertEqual(config.dataset, dataset_list)
        self.assertEqual(config.dataset_kwargs, {})


if __name__ == "__main__":
    unittest.main()
