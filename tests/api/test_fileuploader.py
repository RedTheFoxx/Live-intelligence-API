import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import requests # For requests.exceptions.HTTPError and requests.exceptions.RequestException

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import FileUploader

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TestFileUploader(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key_upload"
        self.base_url = "http://test.paradigm.com"
        self.default_model = "upload-default"
        self.uploader = FileUploader(self.api_key, self.base_url, self.default_model)

    @patch.object(FileUploader, '_make_request')
    def test_open_session_no_param(self, mock_make_request):
        mock_make_request.return_value = {"uuid": "session-uuid-123"}
        result = self.uploader.open_session()
        mock_make_request.assert_called_once_with("POST", "/api/v2/upload-session", data=None)
        self.assertEqual(result, "session-uuid-123")

    @patch.object(FileUploader, '_make_request')
    def test_open_session_with_pipeline(self, mock_make_request):
        mock_make_request.return_value = {"uuid": "session-uuid-456"}
        result = self.uploader.open_session(ingestion_pipeline="v2.2.1")
        mock_make_request.assert_called_once_with("POST", "/api/v2/upload-session", data={"ingestion_pipeline": "v2.2.1"})
        self.assertEqual(result, "session-uuid-456")

    @patch.object(FileUploader, '_make_request')
    def test_open_session_no_uuid(self, mock_make_request):
        mock_make_request.return_value = {"not_uuid": "nope"}
        with self.assertRaisesRegex(ValueError, "Failed to open upload session or UUID not found"):
            self.uploader.open_session()

    @patch("builtins.open", create=True)
    @patch.object(FileUploader, '_make_request')
    def test_upload_file_to_session_success(self, mock_make_request, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_make_request.return_value = {"status": "ok"}
        result = self.uploader.upload_file_to_session(
            session_uuid="uuid-1",
            file_path="/tmp/test.pdf",
            title="Titre",
            metadata_filename="meta.pdf",
            collection_type="private",
            workspace_id=42,
            chunk_size=1000,
            chunk_overlap=0.1,
            ocr_agent="tesseract",
            ocr_complete_doc_table_extraction=True,
            ocr_hi_res_model_name="ocr-model",
            ocr_strategy="fast",
            ocr_timeout=60,
            ocr_url="http://ocr",
            is_visual_ingestion_on=True,
            default_pipeline="v2.2.1"
        )
        expected_fields = {
            "title": "Titre",
            "filename": "meta.pdf",
            "collection_type": "private",
            "workspace_id": 42,
            "chunk_size": 1000,
            "chunk_overlap": 0.1,
            "OCR_AGENT": "tesseract",
            "OCR_COMPLETE_DOC_TABLE_EXTRACTION": True,
            "OCR_HI_RES_MODEL_NAME": "ocr-model",
            "OCR_STRATEGY": "fast",
            "OCR_TIMEOUT": 60,
            "OCR_URL": "http://ocr",
            "IS_VISUAL_INGESTION_ON": True,
            "DEFAULT_PIPELINE": "v2.2.1"
        }
        mock_make_request.assert_called_once()
        args, kwargs = mock_make_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "/api/v2/upload-session/uuid-1")
        self.assertEqual(kwargs["data"], expected_fields)
        self.assertIn("file", kwargs["files"])
        self.assertEqual(result, {"status": "ok"})

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_upload_file_to_session_file_not_found(self, mock_open):
        with self.assertRaisesRegex(ValueError, "File not found at path"): 
            self.uploader.upload_file_to_session("uuid-2", "notfound.pdf")

    def test_upload_file_to_session_workspace_id_required(self):
        # collection_type workspace sans workspace_id
        with self.assertRaisesRegex(ValueError, "workspace_id is required when collection_type is 'workspace'"):
            self.uploader.upload_file_to_session("uuid-3", "file.pdf", collection_type="workspace")

    @patch("builtins.open", create=True)
    @patch.object(FileUploader, '_make_request')
    def test_upload_file_to_session_http_error(self, mock_make_request, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_make_request.side_effect = requests.exceptions.HTTPError("HTTP error")
        with self.assertRaises(requests.exceptions.HTTPError):
            self.uploader.upload_file_to_session("uuid-4", "file.pdf")

    @patch("builtins.open", create=True)
    @patch.object(FileUploader, '_make_request')
    def test_upload_file_to_session_other_exception(self, mock_make_request, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_make_request.side_effect = Exception("Boom!")
        with self.assertRaisesRegex(RuntimeError, "An unexpected error"):
            self.uploader.upload_file_to_session("uuid-5", "file.pdf")

    @patch.object(FileUploader, '_make_request')
    def test_get_session_details(self, mock_make_request):
        mock_make_request.return_value = {"uuid": "session-uuid-6", "status": "active"}
        result = self.uploader.get_session_details("session-uuid-6")
        mock_make_request.assert_called_once_with("GET", "/api/v2/upload-session/session-uuid-6")
        self.assertEqual(result, {"uuid": "session-uuid-6", "status": "active"})

    @patch.object(FileUploader, '_make_request')
    def test_delete_session(self, mock_make_request):
        mock_make_request.return_value = {}
        result = self.uploader.delete_session("session-uuid-7")
        mock_make_request.assert_called_once_with("DELETE", "/api/v2/upload-session/session-uuid-7")
        self.assertEqual(result, {})

    @patch.object(FileUploader, '_make_request')
    def test_deactivate_all_sessions(self, mock_make_request):
        mock_make_request.return_value = {"deactivated": True}
        result = self.uploader.deactivate_all_sessions()
        mock_make_request.assert_called_once_with("POST", "/api/v2/upload-session/deactivate")
        self.assertEqual(result, {"deactivated": True})

    @patch.object(FileUploader, 'upload_file_to_session')
    def test_execute_alias(self, mock_upload):
        mock_upload.return_value = {"uploaded": True}
        result = self.uploader.execute("uuid-8", "file8.pdf", foo=1)
        mock_upload.assert_called_once_with("uuid-8", "file8.pdf", foo=1)
        self.assertEqual(result, {"uploaded": True})

if __name__ == '__main__':
    unittest.main()
