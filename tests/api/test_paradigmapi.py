import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import requests # Should be imported if TestParadigmAPI uses it directly or via ParadigmAPI's _make_request
import urllib3 # For disabling warnings

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import ParadigmAPI # Assuming ParadigmAPI is needed for ConcreteAPI

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define ConcreteAPI at the class level to make it accessible to all methods
class ConcreteAPI(ParadigmAPI):
    def execute(self, *args, **kwargs):
        pass

class TestParadigmAPI(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_key"
        self.base_url = "http://test.paradigm.com"
        self.default_model = "default-model"
        self.concrete_api = ConcreteAPI(self.api_key, self.base_url, self.default_model)


    def test_init(self):
        """Test API initialization."""
        self.assertEqual(self.concrete_api.api_key, self.api_key)
        self.assertEqual(self.concrete_api.base_url, self.base_url)
        self.assertEqual(self.concrete_api.default_model, self.default_model)
        self.assertEqual(self.concrete_api.headers["Authorization"], f"Bearer {self.api_key}")
        self.assertEqual(self.concrete_api.headers["Content-Type"], "application/json")

        api_with_slash = ConcreteAPI(self.api_key, self.base_url + "/", self.default_model)
        self.assertEqual(api_with_slash.base_url, self.base_url)

    @patch('requests.get')
    def test_make_request_get(self, mock_get):
        """Test _make_request for GET method."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        mock_response.content = b'{"data": "success"}' # Ensure content is not empty
        mock_get.return_value = mock_response

        endpoint = "/test_get"
        params = {"param1": "value1"}
        response = self.concrete_api._make_request("GET", endpoint, params=params)

        mock_get.assert_called_once_with(
            f"{self.base_url}{endpoint}",
            headers=self.concrete_api.headers,
            params=params,
            verify=False
        )
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(response, {"data": "success"})

    @patch('requests.post')
    def test_make_request_post(self, mock_post):
        """Test _make_request for POST method."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123}
        mock_response.content = b'{"id": 123}'
        mock_post.return_value = mock_response

        endpoint = "/test_post"
        data = {"key": "value"}
        response = self.concrete_api._make_request("POST", endpoint, data=data)

        mock_post.assert_called_once_with(
            f"{self.base_url}{endpoint}",
            headers=self.concrete_api.headers,
            json=data,
            data=None,
            files=None,
            verify=False
        )
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(response, {"id": 123})

    @patch('requests.put')
    def test_make_request_put(self, mock_put):
        """Test _make_request for PUT method."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"updated": True}
        mock_response.content = b'{"updated": True}'
        mock_put.return_value = mock_response

        endpoint = "/test_put/1"
        data = {"field": "new_value"}
        response = self.concrete_api._make_request("PUT", endpoint, data=data)

        mock_put.assert_called_once_with(
            f"{self.base_url}{endpoint}",
            headers=self.concrete_api.headers,
            json=data,
            data=None,
            files=None,
            verify=False
        )
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(response, {"updated": True})

    @patch('requests.patch')
    def test_make_request_patch(self, mock_patch):
        """Test _make_request for PATCH method."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"patched": True}
        mock_response.content = b'{"patched": True}'
        mock_patch.return_value = mock_response

        endpoint = "/test_patch/1"
        data = {"field": "partial_update"}
        response = self.concrete_api._make_request("PATCH", endpoint, data=data)

        mock_patch.assert_called_once_with(
            f"{self.base_url}{endpoint}",
            headers=self.concrete_api.headers,
            json=data,
            data=None,
            files=None,
            verify=False
        )
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(response, {"patched": True})

    @patch('requests.delete')
    def test_make_request_delete(self, mock_delete):
        """Test _make_request for DELETE method."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b''
        mock_delete.return_value = mock_response

        endpoint = "/test_delete/1"
        response = self.concrete_api._make_request("DELETE", endpoint)

        mock_delete.assert_called_once_with(
            f"{self.base_url}{endpoint}",
            headers=self.concrete_api.headers,
            json=None,
            data=None,
            verify=False
        )
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(response, {})

    @patch('requests.request')
    def test_make_request_unsupported_method(self, mock_request):
        """Test _make_request with an unsupported HTTP method."""
        with self.assertRaisesRegex(ValueError, "Unsupported HTTP method: FOOBAR"):
            self.concrete_api._make_request("FOOBAR", "/test")

    @patch('requests.get')
    def test_make_request_http_error(self, mock_get):
        """Test _make_request handling HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.concrete_api._make_request("GET", "/not_found")
        mock_response.raise_for_status.assert_called_once()

    @patch('requests.get')
    def test_make_request_empty_response_content(self, mock_get):
        """Test _make_request with empty response content (e.g., 204 No Content)."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b'' # Empty content
        mock_get.return_value = mock_response

        response = self.concrete_api._make_request("GET", "/empty")

        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(response, {}) # Should return an empty dict

    def test_ensure_model(self):
        """Test _ensure_model logic."""
        # Model provided
        self.assertEqual(self.concrete_api._ensure_model("specific-model"), "specific-model")
        # No model provided, use default
        self.assertEqual(self.concrete_api._ensure_model(), self.default_model)
        # No model provided, no default set
        api_no_default = ConcreteAPI(self.api_key, self.base_url)
        api_no_default.default_model = None
        with self.assertRaisesRegex(ValueError, "No model specified and no default model set"):
            api_no_default._ensure_model()

    def test_abstract_method_execute(self):
        """Test that execute is an abstract method."""
        with self.assertRaises(TypeError):
            # Cannot instantiate the abstract class directly without implementing execute
            ParadigmAPI(self.api_key, self.base_url)

if __name__ == '__main__':
    unittest.main()
