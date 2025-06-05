import os
import sys
import pytest
import asyncio
import aiohttp # Used by the tests

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from api.live_api import AsyncDocumentSearch

# urllib3 warnings might not be relevant here if not using requests library directly
# but ParadigmAPI base might still initialize it. For safety, let's keep it.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Note: aioresponses is imported within the test methods that use it.

class TestAsyncDocumentSearch:
    """Tests unitaires pour AsyncDocumentSearch (squelette, à compléter)."""
    def setup_method(self):
        self.api_key = "test_key_async"
        self.base_url = "http://test.async.com"
        self.default_model = "async-default"
        self.async_api = AsyncDocumentSearch(self.api_key, self.base_url, self.default_model)

    @pytest.mark.asyncio
    async def test_ensure_model(self):
        assert self.async_api._ensure_model("foo") == "foo"
        assert self.async_api._ensure_model() == self.default_model
        api_no_default = AsyncDocumentSearch(self.api_key, self.base_url)
        api_no_default.default_model = None
        with pytest.raises(ValueError):
            api_no_default._ensure_model()

    @pytest.mark.asyncio
    async def test_execute_single_success(self):
        from aioresponses import aioresponses
        async_api = self.async_api
        query = "test query"
        expected_result = {"results": ["ok"]}
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        async with aioresponses() as m:
            m.post(endpoint, payload=expected_result)
            async with aiohttp.ClientSession() as session:
                result = await async_api.execute_single(session, query=query)
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_execute_single_http_error(self):
        from aioresponses import aioresponses
        async_api = self.async_api
        query = "test query"
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        async with aioresponses() as m:
            m.post(endpoint, status=500)
            async with aiohttp.ClientSession() as session:
                with pytest.raises(Exception) as exc:
                    await async_api.execute_single(session, query=query)
        # Check for either specific or generic part of the exception message
        assert "HTTP request failed" in str(exc.value) or "Request processing failed" in str(exc.value)


    @pytest.mark.asyncio
    async def test_execute_batch_success(self):
        from aioresponses import aioresponses
        async_api = self.async_api
        requests_data = [
            {"query": "q1"},
            {"query": "q2"}
        ]
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        expected1 = {"results": [1]}
        expected2 = {"results": [2]}
        async with aioresponses() as m:
            # Ensure both possible responses are mocked if order isn't guaranteed
            m.post(endpoint, payload=expected1, repeat=True) # repeat=True if calls are identical
            m.post(endpoint, payload=expected2, repeat=True) # Or add more specific mocking
            results = await async_api.execute_batch(requests_data, max_concurrent=2)
        assert len(results) == 2
        # Check results content without assuming order
        result_payloads = [r["result"] for r in results if r["result"] is not None]
        assert expected1 in result_payloads
        assert expected2 in result_payloads
        for r in results:
            assert r["error"] is None


    @pytest.mark.asyncio
    async def test_execute_batch_with_error(self):
        from aioresponses import aioresponses
        async_api = self.async_api
        requests_data = [
            {"query": "q1"}, # Should succeed
            {"query": "q2"}  # Should fail
        ]
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        expected1 = {"results": [1]}
        
        async with aioresponses() as m:
            m.post(endpoint, payload=expected1) # For q1
            m.post(endpoint, status=500)       # For q2
            results = await async_api.execute_batch(requests_data, max_concurrent=2)
            
        assert len(results) == 2
        
        success_result = next((r for r in results if r["result"] == expected1), None)
        error_result = next((r for r in results if r["error"] is not None), None)

        assert success_result is not None
        assert error_result is not None
        assert "HTTP request failed" in str(error_result["error"]) or "Request processing failed" in str(error_result["error"])


    @pytest.mark.asyncio
    async def test_execute_batch_progress_callback(self):
        from aioresponses import aioresponses
        async_api = self.async_api
        requests_data = [
            {"query": "q1"},
            {"query": "q2"},
            {"query": "q3"}
        ]
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        expected = {"results": ["ok"]}
        progress_calls = []
        def progress_callback(done, total):
            progress_calls.append((done, total))
        
        async with aioresponses() as m:
            m.post(endpoint, payload=expected, repeat=True) # Mock for all 3 calls
            await async_api.execute_batch(requests_data, max_concurrent=2, progress_callback=progress_callback)
        
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]

# It's good practice to have a main guard if these tests might be run directly,
# though pytest doesn't require it.
if __name__ == '__main__':
    pytest.main()
