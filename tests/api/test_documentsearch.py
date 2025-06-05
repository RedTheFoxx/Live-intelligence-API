import unittest
from unittest.mock import patch, MagicMock # MagicMock is not used by this class, but good to keep the template consistent
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import DocumentSearch

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestDocumentSearch(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key"
        self.base_url = "http://test.paradigm.com"
        self.default_model = "search-default"
        self.doc_search_api = DocumentSearch(self.api_key, self.base_url, self.default_model)

    @patch.object(DocumentSearch, '_make_request')
    def test_execute_basic(self, mock_make_request):
        """Test DocumentSearch execute with minimal arguments."""
        mock_make_request.return_value = {"results": []}
        query = "search for this"

        response = self.doc_search_api.execute(query)

        expected_data = {
            "query": query,
            "tool": "DocumentSearch", # Default tool
            # No model expected as default is handled by ParadigmAPI base class (not explicitly passed here)
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/chat/document-search", expected_data)
        self.assertEqual(response, {"results": []})

    @patch.object(DocumentSearch, '_make_request')
    def test_execute_all_args(self, mock_make_request):
        """Test DocumentSearch execute with all arguments."""
        mock_make_request.return_value = {"results": ["doc1"]}
        query = "detailed search"
        model = "search-model-override"
        workspace_ids = [1, 2]
        file_ids = [101, 102]
        chat_session_id = 999
        company_scope = True
        private_scope = False
        tool = "VisionDocumentSearch"

        response = self.doc_search_api.execute(
            query,
            model=model,
            workspace_ids=workspace_ids,
            file_ids=file_ids,
            chat_session_id=chat_session_id,
            company_scope=company_scope,
            private_scope=private_scope,
            tool=tool
        )

        expected_data = {
            "query": query,
            "model": model,
            "workspace_ids": workspace_ids,
            "file_ids": file_ids,
            "chat_session_id": chat_session_id,
            "company_scope": company_scope,
            "private_scope": private_scope,
            "tool": tool
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/chat/document-search", expected_data)
        self.assertEqual(response, {"results": ["doc1"]})

    @patch.object(DocumentSearch, '_make_request')
    def test_execute_boolean_args_false(self, mock_make_request):
        """Test DocumentSearch execute with boolean args explicitly False."""
        mock_make_request.return_value = {"results": []}
        query = "boolean test"
        company_scope = False
        private_scope = False

        self.doc_search_api.execute(
            query,
            company_scope=company_scope,
            private_scope=private_scope
        )

        expected_data = {
            "query": query,
            "company_scope": False, # Explicitly False
            "private_scope": False, # Explicitly False
            "tool": "DocumentSearch"
        }
        mock_make_request.assert_called_once_with("POST", "/api/v2/chat/document-search", expected_data)

if __name__ == '__main__':
    unittest.main()
