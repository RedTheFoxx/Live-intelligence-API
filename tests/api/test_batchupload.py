import unittest
from unittest.mock import patch, MagicMock # MagicMock may not be used directly but good to have if base class methods use it
import os
import sys
import time # For patching time.sleep and time.time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import FileUploader # TestBatchUpload tests methods of FileUploader

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Ajout : Tests batch upload (fusion de test_batch_upload.py) ===
class TestBatchUpload(unittest.TestCase):
    """Test cases for the batch upload functionality (fusionné)."""

    def setUp(self):
        self.api_key = "test_key"
        self.base_url = "http://test.example.com"
        self.uploader = FileUploader(self.api_key, self.base_url)

    @patch.object(FileUploader, 'get_session_details')
    def test_wait_for_session_completion_success(self, mock_get_details):
        mock_get_details.return_value = {
            "documents": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "completed"}
            ]
        }
        result = self.uploader._wait_for_session_completion(
            session_uuid="test-uuid",
            max_wait_time=60,
            check_interval=1
        )
        self.assertTrue(result)
        mock_get_details.assert_called_with("test-uuid")

    @patch.object(FileUploader, 'get_session_details')
    @patch('time.sleep')
    def test_wait_for_session_completion_pending(self, mock_sleep, mock_get_details):
        mock_get_details.side_effect = [
            {"documents": [
                {"id": 1, "status": "pending"},
                {"id": 2, "status": "completed"}
            ]},
            {"documents": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "completed"}
            ]}
        ]
        result = self.uploader._wait_for_session_completion(
            session_uuid="test-uuid",
            max_wait_time=60,
            check_interval=1
        )
        self.assertTrue(result)
        self.assertEqual(mock_get_details.call_count, 2)
        mock_sleep.assert_called_with(1)

    @patch.object(FileUploader, 'get_session_details')
    @patch('time.sleep')
    @patch('time.time')
    def test_wait_for_session_completion_timeout(self, mock_time, mock_sleep, mock_get_details):
        mock_time.side_effect = [0, 0, 70] # Simulate time passing to exceed max_wait_time
        mock_get_details.return_value = {"documents": [{"id": 1, "status": "pending"}]}
        result = self.uploader._wait_for_session_completion(
            session_uuid="test-uuid",
            max_wait_time=60,
            check_interval=1
        )
        self.assertFalse(result)

    @patch.object(FileUploader, 'open_session')
    @patch.object(FileUploader, 'upload_file_to_session')
    @patch.object(FileUploader, '_wait_for_session_completion')
    @patch.object(FileUploader, 'deactivate_all_sessions')
    @patch('time.sleep') # To control sleep duration in tests
    def test_upload_files_in_batches_success(self, mock_sleep, mock_deactivate, mock_wait, mock_upload, mock_open):
        mock_open.return_value = "session-uuid-123"
        mock_upload.return_value = {"status": "success"}
        mock_wait.return_value = True
        mock_deactivate.return_value = {} # Assuming it returns a dict upon success
        file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
        result = self.uploader.upload_files_in_batches(
            file_paths=file_paths,
            batch_size=2
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_files"], 3)
        self.assertEqual(result["successful_uploads"], 3)
        self.assertEqual(result["failed_uploads"], 0)
        self.assertEqual(result["batches_processed"], 2)
        self.assertEqual(result["success_rate"], 100.0)
        self.assertEqual(mock_open.call_count, 2)
        self.assertEqual(mock_upload.call_count, 3)
        self.assertEqual(mock_wait.call_count, 2)
        self.assertEqual(mock_deactivate.call_count, 2)

    @patch.object(FileUploader, 'open_session')
    @patch.object(FileUploader, 'upload_file_to_session')
    @patch.object(FileUploader, '_wait_for_session_completion')
    @patch.object(FileUploader, 'deactivate_all_sessions')
    def test_upload_files_in_batches_upload_failure(self, mock_deactivate, mock_wait, mock_upload, mock_open):
        mock_open.return_value = "session-uuid-123"
        mock_wait.return_value = True
        mock_deactivate.return_value = {}
        mock_upload.side_effect = [
            {"status": "success"},
            Exception("Upload failed"), # Simulate failure for the second file
            {"status": "success"}
        ]
        file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
        result = self.uploader.upload_files_in_batches(
            file_paths=file_paths,
            batch_size=10 # Large enough for one batch attempt
        )
        self.assertEqual(result["status"], "completed") # Or "partial_failure" depending on desired status
        self.assertEqual(result["total_files"], 3)
        self.assertEqual(result["successful_uploads"], 2)
        self.assertEqual(result["failed_uploads"], 1)
        self.assertAlmostEqual(result["success_rate"], (2/3)*100, places=1)

    def test_upload_files_in_batches_empty_list(self):
        result = self.uploader.upload_files_in_batches(file_paths=[])
        self.assertEqual(result["status"], "completed") # Or "no_files"
        self.assertEqual(result["total_files"], 0)
        self.assertEqual(result["successful_uploads"], 0)
        self.assertEqual(result["failed_uploads"], 0)
        self.assertEqual(result["batches_processed"], 0)

    def test_upload_files_in_batches_batch_size_limit(self):
        # This test ensures that batching logic correctly creates multiple batches
        # when the number of files exceeds the batch_size_limit (which is 10 internally)
        with patch.object(self.uploader, 'open_session') as mock_open:
            mock_open.return_value = "session-uuid"
            with patch.object(self.uploader, 'upload_file_to_session') as mock_upload:
                mock_upload.return_value = {"status": "success"}
                with patch.object(self.uploader, '_wait_for_session_completion') as mock_wait:
                    mock_wait.return_value = True
                    with patch.object(self.uploader, 'deactivate_all_sessions'):
                        file_paths = [f"file{i}.pdf" for i in range(15)] # More than 10 files
                        result = self.uploader.upload_files_in_batches(
                            file_paths=file_paths,
                            batch_size=15 # User-set batch size, but internal limit is 10
                        )
                        # Expecting 2 batches: one of 10, one of 5 due to internal limit
                        self.assertEqual(result["batches_processed"], 2)


    @patch.object(FileUploader, 'upload_files_in_batches')
    def test_upload_files_to_personal_space(self, mock_batch_upload):
        mock_batch_upload.return_value = {"status": "completed"}
        file_paths = ["file1.pdf", "file2.pdf"]
        result = self.uploader.upload_files_to_personal_space(
            file_paths=file_paths,
            collection_type="private" # This is the key part for this method
        )
        mock_batch_upload.assert_called_once_with(
            file_paths=file_paths,
            collection_type="private" # Ensure this is passed through
        )
        self.assertEqual(result["status"], "completed")

    def test_upload_files_to_personal_space_workspace_validation(self):
        file_paths = ["file1.pdf"]
        with self.assertRaises(ValueError) as context:
            self.uploader.upload_files_to_personal_space(
                file_paths=file_paths,
                collection_type="workspace" # Invalid for this specific helper
            )
        self.assertIn("workspace_id is required", str(context.exception))

if __name__ == '__main__':
    unittest.main()
