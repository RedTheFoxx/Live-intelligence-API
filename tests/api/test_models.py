import unittest
from unittest.mock import patch
import os
import sys
import requests # For requests.exceptions.RequestException

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import Models

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestModels(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key_models"
        self.base_url = "http://test.paradigm.com"
        # No default model needed for this endpoint
        self.models_api = Models(self.api_key, self.base_url)

    @patch.object(Models, '_make_request')
    def test_execute_success(self, mock_make_request):
        """Test Models execute successfully fetching technical names."""
        mock_response_data = {
            "data": [
                {"id": 1, "name": "Model One", "technical_name": "model-one-v1"},
                {"id": 2, "name": "Model Two", "technical_name": "model-two-alpha"},
                {"id": 3, "name": "Model Three", "technical_name": "model-three"},
                {"id": 4, "name": "No Tech Name", "some_other_field": "value"}, # Should be skipped
                "not a dict" # Should be skipped
            ]
        }
        mock_make_request.return_value = mock_response_data

        result = self.models_api.execute()

        mock_make_request.assert_called_once_with("GET", "/api/v2/models")
        expected_names = ["model-one-v1", "model-two-alpha", "model-three"]
        self.assertListEqual(result, expected_names)

    @patch.object(Models, '_make_request')
    def test_execute_invalid_structure_no_data(self, mock_make_request):
        """Test Models execute with invalid response structure (missing 'data')."""
        mock_make_request.return_value = {"message": "error"} # Missing 'data' key

        with self.assertRaisesRegex(ValueError, "Invalid response structure received from API"):
            self.models_api.execute()
        mock_make_request.assert_called_once_with("GET", "/api/v2/models")

    @patch.object(Models, '_make_request')
    def test_execute_invalid_structure_data_not_list(self, mock_make_request):
        """Test Models execute with invalid response structure ('data' is not a list)."""
        mock_make_request.return_value = {"data": {"key": "value"}} # 'data' is a dict

        with self.assertRaisesRegex(ValueError, "Invalid response structure received from API"):
            self.models_api.execute()
        mock_make_request.assert_called_once_with("GET", "/api/v2/models")

    @patch.object(Models, '_make_request')
    def test_execute_api_request_fails(self, mock_make_request):
        """Test Models execute when the API request itself fails."""
        mock_make_request.side_effect = requests.exceptions.RequestException("Connection error")

        with self.assertRaises(requests.exceptions.RequestException):
            self.models_api.execute()
        mock_make_request.assert_called_once_with("GET", "/api/v2/models")

if __name__ == '__main__':
    unittest.main()
