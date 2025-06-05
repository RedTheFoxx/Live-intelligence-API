import unittest
from unittest.mock import patch
import os
import sys
import requests # For requests.exceptions.RequestException

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import Files

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestFiles(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key_files"
        self.base_url = "http://test.paradigm.com"
        # No default model needed
        self.files_api = Files(self.api_key, self.base_url)

    @patch.object(Files, '_make_request')
    def test_execute_no_params(self, mock_make_request):
        """Test Files execute with no optional parameters."""
        mock_response_data = {
            "data": [
                {"id": 10, "filename": "file1.pdf", "size": 1024},
                {"id": 20, "filename": "document2.docx", "created_at": "..."},
                {"id": 30}, # Missing filename, should be skipped
                {"filename": "file4.txt"}, # Missing id, should be skipped
                {"id": 50, "filename": "report5.txt"}
            ]
        }
        mock_make_request.return_value = mock_response_data

        result = self.files_api.execute()

        # Expect call with empty params dict
        mock_make_request.assert_called_once_with("GET", "/api/v2/files", params={})
        expected_files = [
            {"id": 10, "filename": "file1.pdf"},
            {"id": 20, "filename": "document2.docx"},
            {"id": 50, "filename": "report5.txt"}
        ]
        self.assertListEqual(result, expected_files)

    @patch.object(Files, '_make_request')
    def test_execute_with_params(self, mock_make_request):
        """Test Files execute with all optional parameters."""
        mock_make_request.return_value = {"data": []} # Response content doesn't matter for this test

        company_scope = True
        private_scope = False
        workspace_scope = 123
        page = 2

        self.files_api.execute(
            company_scope=company_scope,
            private_scope=private_scope,
            workspace_scope=workspace_scope,
            page=page
        )

        expected_params = {
            "company_scope": True,
            "private_scope": False,
            "workspace_scope": 123,
            "page": 2
        }
        mock_make_request.assert_called_once_with("GET", "/api/v2/files", params=expected_params)

    @patch.object(Files, '_make_request')
    def test_execute_only_some_params(self, mock_make_request):
        """Test Files execute with a subset of optional parameters."""
        mock_make_request.return_value = {"data": []}

        private_scope = True
        page = 5

        self.files_api.execute(private_scope=private_scope, page=page)

        expected_params = {
            "private_scope": True,
            "page": 5
        }
        mock_make_request.assert_called_once_with("GET", "/api/v2/files", params=expected_params)

    @patch.object(Files, '_make_request')
    def test_execute_empty_response(self, mock_make_request):
        """Test Files execute with an empty data list in the response."""
        mock_make_request.return_value = {"data": []}

        result = self.files_api.execute()

        mock_make_request.assert_called_once_with("GET", "/api/v2/files", params={})
        self.assertListEqual(result, [])

    @patch.object(Files, '_make_request')
    def test_execute_api_request_fails(self, mock_make_request):
        """Test Files execute when the API request itself fails."""
        mock_make_request.side_effect = requests.exceptions.RequestException("Timeout")

        with self.assertRaises(requests.exceptions.RequestException):
            self.files_api.execute()
        mock_make_request.assert_called_once_with("GET", "/api/v2/files", params={})

if __name__ == '__main__':
    unittest.main()
