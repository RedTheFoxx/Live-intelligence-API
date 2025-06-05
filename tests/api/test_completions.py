import unittest
from unittest.mock import patch, MagicMock # Ensure MagicMock is imported if used, e.g. for mock_response
import os
import sys
import requests # For potential patching of requests.post

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import Completions

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestCompletions(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key"
        self.base_url = "http://test.paradigm.com"
        self.default_model = "compl-default"
        self.compl_api = Completions(self.api_key, self.base_url, self.default_model)

    @patch.object(Completions, '_make_request')
    @patch.object(Completions, '_ensure_model')
    def test_execute(self, mock_ensure_model, mock_make_request):
        """Test Completions execute method."""
        model_to_use = "text-davinci-003"
        mock_ensure_model.return_value = model_to_use
        mock_make_request.return_value = {"choices": [{"text": "completion text"}]}

        prompt = "Once upon a time"
        kwargs = {"max_tokens": 50}
        response = self.compl_api.execute(prompt, model="override-compl-model", **kwargs)

        mock_ensure_model.assert_called_once_with("override-compl-model")
        expected_data = {
            "model": model_to_use,
            "prompt": prompt,
            "max_tokens": 50
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/completions", expected_data)
        self.assertEqual(response, {"choices": [{"text": "completion text"}]})

    @patch.object(Completions, '_make_request')
    def test_execute_uses_default_model(self, mock_make_request):
        """Test Completions execute uses default model when none is provided."""
        mock_make_request.return_value = {"choices": [{"text": "default completion"}]}
        prompt = "To be or not to be"

        self.compl_api.execute(prompt) # No model specified

        expected_data = {
            "model": self.default_model, # Should use the default
            "prompt": prompt,
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/completions", expected_data)

    @patch('requests.post')
    @patch.object(Completions, '_ensure_model')
    def test_stream_basic(self, mock_ensure_model, mock_post):
        """Test Completions.stream yields chunks and handles [DONE] and usage stats."""
        mock_ensure_model.return_value = self.default_model
        lines = [
            b'data: {"choices": [{"text": "foo"}]}',
            b'data: {"choices": [{"text": "bar"}]}',
            b'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}',
            b'data: [DONE]'
        ]
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response
        prompt = "Hello"
        # Re-instantiate compl_api for this specific stream test if it modifies internal state
        # or ensure setUp is sufficient. For this test, it seems okay.
        stream = self.compl_api.stream(prompt)
        results = list(stream)
        self.assertEqual(results[0]["choices"][0]["text"], "foo")
        self.assertEqual(results[1]["choices"][0]["text"], "bar")
        self.assertEqual(results[2]["usage"], {"prompt_tokens": 3, "completion_tokens": 1})

    @patch('requests.post')
    @patch.object(Completions, '_ensure_model')
    def test_stream_json_error(self, mock_ensure_model, mock_post):
        """Test Completions.stream yields error dict on JSON decode error."""
        mock_ensure_model.return_value = self.default_model
        lines = [b'data: {not a valid json}', b'data: [DONE]']
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response
        prompt = "Hello"
        # Similar to above, ensure compl_api state is fine or re-instantiate.
        stream = self.compl_api.stream(prompt)
        result = next(stream)
        self.assertIn("error", result)
        self.assertIn("raw", result)

if __name__ == '__main__':
    unittest.main()
