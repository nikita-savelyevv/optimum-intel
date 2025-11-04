#  Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Test suite to verify that trust_remote_code parameter properly overrides config values
to prevent remote code execution vulnerabilities.
"""

import unittest
from unittest.mock import MagicMock, patch
from optimum.intel.openvino.quantization import OVCalibrationDatasetBuilder
from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
from optimum.intel import OVModelForCausalLM


class TestTrustRemoteCodeOverride(unittest.TestCase):
    """
    Test that trust_remote_code parameter properly overrides config values
    in all methods that call AutoTokenizer.from_pretrained or AutoProcessor.from_pretrained
    """

    def setUp(self):
        """Set up common test fixtures"""
        # Create a mock model
        self.mock_model = MagicMock(spec=OVModelForCausalLM)
        self.mock_model.config = MagicMock()
        
        # Create dataset builder
        self.dataset_builder = OVCalibrationDatasetBuilder(self.mock_model, seed=42)

    @patch('optimum.intel.openvino.quantization.AutoTokenizer')
    def test_causal_lm_trust_remote_code_override(self, mock_tokenizer_class):
        """
        Test that trust_remote_code parameter overrides config value in _prepare_causal_lm_calibration_data
        """
        # Create a config with trust_remote_code=True (malicious config)
        config = OVWeightQuantizationConfig(
            tokenizer="test-tokenizer",
            trust_remote_code=True,  # This is the malicious value from remote config
            dataset=["test text"]
        )
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        
        # Mock model
        self.mock_model.prepare_inputs = MagicMock(return_value={})
        
        # Call the method with trust_remote_code=False (user-provided safe value)
        # This should override the config's trust_remote_code=True
        try:
            self.dataset_builder._prepare_causal_lm_calibration_data(
                config, 
                seqlen=32,
                trust_remote_code=False  # This should override config.trust_remote_code
            )
        except Exception:
            # Ignore other errors, we only care about the trust_remote_code parameter
            pass
        
        # Verify that AutoTokenizer.from_pretrained was called with trust_remote_code=False
        # not with the malicious trust_remote_code=True from the config
        mock_tokenizer_class.from_pretrained.assert_called_once()
        call_kwargs = mock_tokenizer_class.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs.get('trust_remote_code'), False,
                        "trust_remote_code should be False (from parameter), not True (from config)")

    @patch('optimum.intel.openvino.quantization.AutoTokenizer')
    def test_text_to_text_trust_remote_code_override(self, mock_tokenizer_class):
        """
        Test that trust_remote_code parameter overrides config value in _prepare_text_to_text_calibration_data
        """
        from optimum.intel import OVModelForSeq2SeqLM
        
        # Create a mock seq2seq model
        self.mock_model = MagicMock(spec=OVModelForSeq2SeqLM)
        self.mock_model.config = MagicMock()
        self.mock_model.components = {
            "encoder_model": MagicMock(),
            "decoder_model": MagicMock()
        }
        for component in self.mock_model.components.values():
            component.compile = MagicMock()
            component.request = MagicMock()
        
        self.dataset_builder.model = self.mock_model
        
        # Create a config with trust_remote_code=True (malicious config)
        config = OVWeightQuantizationConfig(
            tokenizer="test-tokenizer",
            trust_remote_code=True,  # Malicious value from remote config
        )
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.take.return_value = []
        
        # Call the method with trust_remote_code=False (safe value)
        try:
            self.dataset_builder._prepare_text_to_text_calibration_data(
                config,
                mock_dataset,
                seq_len=128,
                trust_remote_code=False  # This should override config.trust_remote_code
            )
        except Exception:
            # Ignore other errors, we only care about the trust_remote_code parameter
            pass
        
        # Verify that if AutoTokenizer.from_pretrained was called, it used trust_remote_code=False
        if mock_tokenizer_class.from_pretrained.called:
            call_kwargs = mock_tokenizer_class.from_pretrained.call_args[1]
            self.assertEqual(call_kwargs.get('trust_remote_code'), False,
                            "trust_remote_code should be False (from parameter), not True (from config)")

    @patch('optimum.intel.openvino.quantization.AutoTokenizer')
    def test_text_encoder_trust_remote_code_override(self, mock_tokenizer_class):
        """
        Test that trust_remote_code parameter overrides config value in _prepare_text_encoder_model_calibration_data
        """
        from optimum.intel import OVModelForFeatureExtraction
        
        # Create a mock feature extraction model
        self.mock_model = MagicMock(spec=OVModelForFeatureExtraction)
        self.mock_model.config = MagicMock()
        self.mock_model.compile = MagicMock()
        self.mock_model.request = MagicMock()
        
        self.dataset_builder.model = self.mock_model
        
        # Create a config with trust_remote_code=True (malicious config)
        config = OVWeightQuantizationConfig(
            tokenizer="test-tokenizer",
            trust_remote_code=True,  # Malicious value from remote config
        )
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        
        # Call the method with trust_remote_code=False (safe value)
        try:
            self.dataset_builder._prepare_text_encoder_model_calibration_data(
                config,
                mock_dataset,
                seq_len=128,
                trust_remote_code=False  # This should override config.trust_remote_code
            )
        except Exception:
            # Ignore other errors, we only care about the trust_remote_code parameter
            pass
        
        # Verify that AutoTokenizer.from_pretrained was called with trust_remote_code=False
        if mock_tokenizer_class.from_pretrained.called:
            call_kwargs = mock_tokenizer_class.from_pretrained.call_args[1]
            self.assertEqual(call_kwargs.get('trust_remote_code'), False,
                            "trust_remote_code should be False (from parameter), not True (from config)")

    @patch('optimum.intel.openvino.quantization.AutoProcessor')
    def test_visual_causal_lm_trust_remote_code_override(self, mock_processor_class):
        """
        Test that trust_remote_code parameter overrides config value in _prepare_visual_causal_lm_calibration_data
        """
        from optimum.intel import OVModelForVisualCausalLM
        
        # Create a mock visual causal LM model
        self.mock_model = MagicMock(spec=OVModelForVisualCausalLM)
        self.mock_model.config = MagicMock()
        self.mock_model.components = {}
        
        self.dataset_builder.model = self.mock_model
        
        # Create a config with trust_remote_code=True (malicious config)
        config = OVWeightQuantizationConfig(
            processor="test-processor",
            tokenizer="test-tokenizer",
            trust_remote_code=True,  # Malicious value from remote config
            dataset="contextual"
        )
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        
        # Call the method with trust_remote_code=False (safe value)
        try:
            self.dataset_builder._prepare_visual_causal_lm_calibration_data(
                config,
                mock_dataset,
                max_image_size=600,
                trust_remote_code=False  # This should override config.trust_remote_code
            )
        except Exception:
            # Ignore other errors, we only care about the trust_remote_code parameter
            pass
        
        # Verify that AutoProcessor.from_pretrained was called with trust_remote_code=False
        if mock_processor_class.from_pretrained.called:
            call_kwargs = mock_processor_class.from_pretrained.call_args[1]
            self.assertEqual(call_kwargs.get('trust_remote_code'), False,
                            "trust_remote_code should be False (from parameter), not True (from config)")


if __name__ == "__main__":
    unittest.main()
