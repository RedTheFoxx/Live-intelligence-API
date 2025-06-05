import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# DocumentSearchAlt uses ChatCompletions internally, so that needs to be available for patching.
# It also directly uses DocumentSearchAlt from api.live_api
from api.live_api import DocumentSearchAlt, ChatCompletions

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestDocumentSearchAlt(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key_alt"
        self.base_url = "http://test.paradigm.com"
        self.default_model = "alt-search-default"
        self.doc_search_alt_api = DocumentSearchAlt(self.api_key, self.base_url, self.default_model)

    @patch.object(DocumentSearchAlt, '_make_request')
    def test_query_documents(self, mock_make_request):
        """Test the _query_documents helper method."""
        mock_make_request.return_value = [{"chunks": [{"text": "chunk1"}]}]
        query = "find chunks"
        n = 10

        result = self.doc_search_alt_api._query_documents(query, n=n)

        expected_data = {"query": query, "n": n}
        mock_make_request.assert_called_once_with("POST", "/api/v2/query", expected_data)
        self.assertEqual(result, [{"chunks": [{"text": "chunk1"}]}])

    @patch('api.live_api.ChatCompletions') # Patch ChatCompletions class used internally
    def test_generate_response(self, MockChatCompletions):
        """Test the _generate_response helper method."""
        # Setup mock ChatCompletions instance and its execute method
        mock_chat_instance = MagicMock()
        mock_chat_execute_response = {"choices": [{"message": {"content": "Generated response"}}]}
        mock_chat_instance.execute.return_value = mock_chat_execute_response
        MockChatCompletions.return_value = mock_chat_instance

        query = "Summarize this"
        # Simulate the structure returned by _query_documents
        context_chunks = [
             {
                "chunks": [
                    {"text": "First piece of context.", "metadata": {"source": "doc1.pdf"}},
                    {"text": "Second piece.", "metadata": {"source": "doc2.txt"}}
                ]
             },
             { # Simulate another potential set of chunks (e.g., from another source/query variation)
                 "chunks": [
                     {"text": "Third relevant sentence.", "metadata": {"source": "doc1.pdf"}}
                 ]
             }
        ]
        model = "gen-model"

        response, messages = self.doc_search_alt_api._generate_response(query, context_chunks, model)

        # Verify ChatCompletions was instantiated correctly
        MockChatCompletions.assert_called_once_with(self.api_key, self.base_url, self.default_model)

        # Verify context formatting
        expected_context = """--- Start Chunk 1 (Source: doc1.pdf) ---
First piece of context.
--- End Chunk 1 ---

--- Start Chunk 2 (Source: doc2.txt) ---
Second piece.
--- End Chunk 2 ---

--- Start Chunk 3 (Source: doc1.pdf) ---
Third relevant sentence.
--- End Chunk 3 ---
"""

        # Verify messages structure
        expected_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the user's question. "
                          "Only use information from the provided context. If you cannot answer based on the context, "
                          "say so."
            },
            {
                "role": "user",
                "content": f"""QUESTION OF USER : {query}\n\nCONTEXT TO USE : {expected_context}"""
            }
        ]

        # Verify chat execute call
        mock_chat_instance.execute.assert_called_once_with(expected_messages, model)

        # Verify results
        self.assertEqual(response, mock_chat_execute_response)
        self.assertEqual(messages, expected_messages)

    @patch.object(DocumentSearchAlt, '_query_documents')
    @patch.object(DocumentSearchAlt, '_generate_response')
    def test_execute_success(self, mock_generate_response, mock_query_documents):
        """Test DocumentSearchAlt execute method successfully."""
        query = "alt search query"
        model = "alt-model-override"
        mock_chunks = [{"chunks": [{"text": "chunk data"}]}]
        mock_gen_response = {"choices": [{"message": {"content": "final answer"}}]}
        mock_prompt_messages = [{"role": "user", "content": "..."}]

        mock_query_documents.return_value = mock_chunks
        mock_generate_response.return_value = (mock_gen_response, mock_prompt_messages)

        result = self.doc_search_alt_api.execute(query, model=model)

        mock_query_documents.assert_called_once_with(query)
        mock_generate_response.assert_called_once_with(query, mock_chunks, model)

        expected_result = {
            "response": mock_gen_response,
            "source_chunks": mock_chunks,
            "prompt_messages": mock_prompt_messages
        }
        self.assertEqual(result, expected_result)

    @patch.object(DocumentSearchAlt, '_query_documents')
    @patch.object(DocumentSearchAlt, '_generate_response')
    def test_execute_no_chunks(self, mock_generate_response, mock_query_documents):
        """Test DocumentSearchAlt execute when no chunks are found."""
        query = "query yielding no chunks"
        mock_query_documents.return_value = [] # Simulate no chunks found

        result = self.doc_search_alt_api.execute(query)

        mock_query_documents.assert_called_once_with(query)
        mock_generate_response.assert_not_called() # Should not attempt generation

        expected_result = {"error": "No relevant content found via query endpoint."}
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
