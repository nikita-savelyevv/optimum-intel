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

import unittest

import pytest

from optimum.intel.openvino.configuration import OVQuantizationConfigBase


class TestDatasetParsing(unittest.TestCase):
    """Test suite for dataset option parsing in OVQuantizationConfigBase."""

    def test_dataset_no_options(self):
        """Test that a simple dataset name without options is preserved."""
        config = OVQuantizationConfigBase(dataset="wikitext")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config.dataset_kwargs, {})

    def test_dataset_with_seq_len_option(self):
        """Test parsing of seq_len option from dataset string."""
        config = OVQuantizationConfigBase(dataset="wikitext:seq_len=128")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 128})

    def test_dataset_gsm8k_with_seq_len(self):
        """Test parsing of seq_len option for gsm8k dataset."""
        config = OVQuantizationConfigBase(dataset="gsm8k:seq_len=512")
        self.assertEqual(config.dataset, "gsm8k")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 512})

    def test_dataset_with_multiple_spaces(self):
        """Test parsing with spaces around the option."""
        config = OVQuantizationConfigBase(dataset="wikitext:seq_len = 64")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config.dataset_kwargs, {"seq_len": 64})

    def test_dataset_list_no_parsing(self):
        """Test that list datasets skip parsing and remain unchanged."""
        dataset_list = ["sample text 1", "sample text 2", "sample text 3"]
        config = OVQuantizationConfigBase(dataset=dataset_list)
        self.assertEqual(config.dataset, dataset_list)
        self.assertEqual(config.dataset_kwargs, {})

    def test_dataset_unsupported_option(self):
        """Test that unsupported options raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:foo=bar")
        assert "Unsupported dataset option 'foo'" in str(exc_info.value)
        assert "Only 'seq_len' is supported" in str(exc_info.value)

    def test_dataset_malformed_option_no_equals(self):
        """Test that options without '=' raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:seq_len")
        assert "Malformed dataset option" in str(exc_info.value)
        assert "Expected format: 'key=value'" in str(exc_info.value)

    def test_dataset_invalid_seq_len_value(self):
        """Test that non-integer seq_len values raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:seq_len=abc")
        assert "Invalid value 'abc' for seq_len" in str(exc_info.value)
        assert "Expected an integer" in str(exc_info.value)

    def test_dataset_empty_string_option(self):
        """Test that empty seq_len value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:seq_len=")
        assert "Invalid value '' for seq_len" in str(exc_info.value)

    def test_dataset_none(self):
        """Test that None dataset is handled correctly."""
        config = OVQuantizationConfigBase(dataset=None)
        self.assertIsNone(config.dataset)
        self.assertEqual(config.dataset_kwargs, {})

    def test_dataset_with_colon_in_name_only(self):
        """Test handling of dataset string with trailing colon but no options."""
        # This should parse, but result in empty options
        config = OVQuantizationConfigBase(dataset="wikitext:")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config.dataset_kwargs, {})


if __name__ == "__main__":
    unittest.main()
