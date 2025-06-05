import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import requests # For potential patching of requests.post if not fully handled by _make_request mock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import ChatCompletions

# If urllib3 was used globally, add it here too, though it might not be strictly necessary for this class
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestChatCompletions(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key"
        self.base_url = "http://test.paradigm.com"
        self.default_model = "chat-default"
        self.chat_api = ChatCompletions(self.api_key, self.base_url, self.default_model)

    @patch.object(ChatCompletions, '_make_request')
    @patch.object(ChatCompletions, '_ensure_model')
    def test_execute(self, mock_ensure_model, mock_make_request):
        """Test ChatCompletions execute method."""
        model_to_use = "gpt-4"
        mock_ensure_model.return_value = model_to_use
        mock_make_request.return_value = {"choices": [{"message": {"content": "response"}}]}

        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"temperature": 0.7}
        response = self.chat_api.execute(messages, model="override-model", **kwargs)

        mock_ensure_model.assert_called_once_with("override-model")
        expected_data = {
            "model": model_to_use,
            "messages": messages,
            "temperature": 0.7
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/chat/completions", expected_data)
        self.assertEqual(response, {"choices": [{"message": {"content": "response"}}]})

    @patch.object(ChatCompletions, '_make_request')
    def test_execute_uses_default_model(self, mock_make_request):
        """Test ChatCompletions execute uses default model when none is provided."""
        mock_make_request.return_value = {"choices": [{"message": {"content": "default response"}}]}
        messages = [{"role": "user", "content": "Hi again"}]

        self.chat_api.execute(messages) # No model specified

        expected_data = {
            "model": self.default_model, # Should use the default
            "messages": messages,
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/chat/completions", expected_data)
        
    @patch.object(ChatCompletions, '_make_request')
    def test_execute_with_history(self, mock_make_request):
        """Test ChatCompletions execute with history parameter."""
        mock_make_request.return_value = {"choices": [{"message": {"content": "response with history"}}]}
        
        history = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "It's sunny today."}
        ]
        messages = [{"role": "user", "content": "And tomorrow?"}]
        
        response = self.chat_api.execute(messages, history=history)
        
        # Check that history and messages are combined correctly
        expected_combined_messages = history + messages
        expected_data = {
            "model": self.default_model,
            "messages": expected_combined_messages,
        }
        
        mock_make_request.assert_called_once_with("POST", "/api/v2/chat/completions", expected_data)
        self.assertEqual(response, {"choices": [{"message": {"content": "response with history"}}]})

    @patch('requests.post')
    @patch.object(ChatCompletions, '_ensure_model')
    def test_stream_basic(self, mock_ensure_model, mock_post):
        """Test ChatCompletions.stream yields chunks and handles [DONE] and usage stats."""
        mock_ensure_model.return_value = self.default_model
        # Simule deux chunks puis [DONE] et usage
        lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}',
            b'data: [DONE]'
        ]
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        stream = self.chat_api.stream(messages)
        results = list(stream)
        # On doit avoir 2 chunks puis un dict usage
        self.assertEqual(results[0]["choices"][0]["delta"]["content"], "Hello")
        self.assertEqual(results[1]["choices"][0]["delta"]["content"], " world")
        self.assertEqual(results[2]["usage"], {"prompt_tokens": 5, "completion_tokens": 2})

    @patch('requests.post')
    @patch.object(ChatCompletions, '_ensure_model')
    def test_stream_json_error(self, mock_ensure_model, mock_post):
        """Test ChatCompletions.stream yields error dict on JSON decode error."""
        mock_ensure_model.return_value = self.default_model
        lines = [b'data: {not a valid json}', b'data: [DONE]']
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response
        messages = [{"role": "user", "content": "Hi"}]
        stream = self.chat_api.stream(messages)
        result = next(stream)
        self.assertIn("error", result)
        self.assertIn("raw", result)

if __name__ == '__main__':
    unittest.main()
