"""Global implementation of Paradigm's API for Live Intelligence and our LLM Quality use-cases."""

import abc
import json
import os
import requests
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

# Deactivate SSL verification for local testing
requests.packages.urllib3.disable_warnings()


class ParadigmAPI(abc.ABC):
    """
    Abstract base class for Paradigm API interactions.
    All API endpoints require a Bearer API key for authentication, a base URL, and most require a model name.
    """

    def __init__(self, api_key: str, base_url: str, default_model: str = None):
        """
        Initialize the API client with authentication, base URL and default model.

        Args:
            api_key (str): The API key for authentication
            base_url (str): The base URL for the API endpoints
            default_model (str, optional): Default model to use for requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self.default_model = default_model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        files: Dict = None,
    ) -> Dict:
        """
        Make an HTTP request to the Paradigm API.

        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint path
            data (Dict, optional): Request payload (JSON for most, form data if 'files' is present)
            params (Dict, optional): Query parameters
            files (Dict, optional): Files to send as multipart/form-data. Key is field name, value is file tuple.

        Returns:
            Dict: API response
        """
        url = f"{self.base_url}{endpoint}"

        request_headers = self.headers.copy()

        json_payload = None
        form_data_payload = None

        if files:  # Multipart request
            if "Content-Type" in request_headers:
                del request_headers[
                    "Content-Type"
                ]  # Let requests set Content-Type for multipart
            form_data_payload = data  # 'data' becomes form fields for multipart
        elif data and method.upper() in [
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        ]:  # JSON request or form data for DELETE
            # For non-file POST/PUT/PATCH, assume JSON unless Content-Type is different
            # For DELETE, data might be JSON or form, depending on API spec. Here, assume JSON if Content-Type is application/json
            if request_headers.get("Content-Type") == "application/json":
                json_payload = data
            else:  # If Content-Type is not explicitly JSON, it might be form-urlencoded for some APIs (not typical for Paradigm here)
                form_data_payload = data

        if method.upper() == "GET":
            response = requests.get(
                url, headers=request_headers, params=params, verify=False
            )
        elif method.upper() == "POST":
            response = requests.post(
                url,
                headers=request_headers,
                json=json_payload,
                data=form_data_payload,
                files=files,
                verify=False,
            )
        elif method.upper() == "PUT":
            response = requests.put(
                url,
                headers=request_headers,
                json=json_payload,
                data=form_data_payload,
                files=files,
                verify=False,
            )
        elif method.upper() == "PATCH":
            response = requests.patch(
                url,
                headers=request_headers,
                json=json_payload,
                data=form_data_payload,
                files=files,
                verify=False,
            )
        elif method.upper() == "DELETE":
            response = requests.delete(
                url,
                headers=request_headers,
                json=json_payload,
                data=form_data_payload,
                verify=False,
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()

        if response.status_code == 204:  # No Content
            return {}

        if response.content:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError as e:
                # If response has content but it's not JSON, this is likely an issue.
                raise ValueError(
                    f"API returned non-JSON response when JSON was expected. "
                    f"Status: {response.status_code}. Content: {response.text[:200]}"
                ) from e
        return {}

    def _ensure_model(self, model: str = None) -> str:
        """
        Ensure a model is specified, using the default if none is provided.

        Args:
            model (str, optional): Model name to use

        Returns:
            str: The model name to use

        Raises:
            ValueError: If no model is specified and no default is set
        """
        model = model or self.default_model
        if not model:
            raise ValueError("No model specified and no default model set")
        return model

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the specific API workflow.
        Must be implemented by subclasses.
        """
        pass


class ChatCompletions(ParadigmAPI):
    """Implementation for chat completions API endpoint."""

    def execute(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        history: List[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict:
        """
        Generate chat completions from a Large Language Model.

        Args:
            messages (List[Dict[str, str]]): List of message objects with role and content
            model (str, optional): Model to use, defaults to the instance default model
            history (List[Dict[str, str]], optional): Previous conversation history to prepend to messages
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dict: API response
        """
        model = self._ensure_model(model)

        # Combine history with new messages if history is provided
        combined_messages = []
        if history:
            combined_messages.extend(history)
        combined_messages.extend(messages)

        data = {"model": model, "messages": combined_messages, **kwargs}

        return self._make_request("POST", "/api/v2/chat/completions", data)

    def stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        history: List[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Stream chat completions from a Large Language Model (OpenAI-compatible streaming).
        Yields each chunk as it is received. At the end, yields a dict with usage stats if present.
        """
        model = self._ensure_model(model)
        combined_messages = []
        if history:
            combined_messages.extend(history)
        combined_messages.extend(messages)
        data = {"model": model, "messages": combined_messages, **kwargs, "stream": True}
        url = f"{self.base_url}/api/v2/chat/completions"
        response = requests.post(
            url, headers=self.headers, json=data, stream=True, verify=False
        )
        usage_stats = None
        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                payload = line[len(b"data: ") :]
                if payload == b"[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                    choices = chunk.get("choices")
                    if (
                        isinstance(choices, list)
                        and len(choices) == 0
                        and "usage" in chunk
                    ):
                        usage_stats = chunk["usage"]
                        continue
                    yield chunk
                except Exception as e:
                    yield {"error": str(e), "raw": payload}
        if usage_stats is not None:
            yield {"usage": usage_stats}


class Completions(ParadigmAPI):
    """Implementation for text completions API endpoint."""

    def execute(self, prompt: str, model: str = None, **kwargs) -> Dict:
        """
        Generate text completions from a Large Language Model.

        Args:
            prompt (str): The prompt to complete
            model (str, optional): Model to use, defaults to the instance default model
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dict: API response
        """
        model = self._ensure_model(model)

        data = {"model": model, "prompt": prompt, **kwargs}

        return self._make_request("POST", "/api/v2/completions", data)

    def stream(self, prompt: str, model: str = None, **kwargs):
        """
        Stream text completions from a Large Language Model (OpenAI-compatible streaming).
        Yields each chunk as it is received. At the end, yields a dict with usage stats if present.
        """
        model = self._ensure_model(model)
        data = {"model": model, "prompt": prompt, **kwargs, "stream": True}
        url = f"{self.base_url}/api/v2/completions"
        response = requests.post(
            url, headers=self.headers, json=data, stream=True, verify=False
        )
        usage_stats = None
        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                payload = line[len(b"data: ") :]
                if payload == b"[DONE]":
                    break
                import json

                try:
                    chunk = json.loads(payload)
                    choices = chunk.get("choices")
                    if (
                        isinstance(choices, list)
                        and len(choices) == 0
                        and "usage" in chunk
                    ):
                        usage_stats = chunk["usage"]
                        continue
                    yield chunk
                except Exception as e:
                    yield {"error": str(e), "raw": payload}
        if usage_stats is not None:
            yield {"usage": usage_stats}


class DocumentSearch(ParadigmAPI):
    """Implementation for document search API endpoint."""

    def execute(
        self,
        query: str,
        model: str = None,
        workspace_ids: List[int] = None,
        file_ids: List[int] = None,
        chat_session_id: int = None,
        company_scope: bool = None,
        private_scope: bool = None,
        tool: str = "DocumentSearch",
    ) -> Dict:
        """
        Search documents and generate responses.

        Args:
            query (str): The search query
            model (str, optional): Model to use, defaults to the instance default model
            workspace_ids (List[int], optional): List of workspace IDs to search
            file_ids (List[int], optional): List of document IDs to search
            chat_session_id (int, optional): Chat session ID for follow-up
            company_scope (bool, optional): Include documents from company collection
            private_scope (bool, optional): Include documents from private collection
            tool (str, optional): Tool to use, either 'DocumentSearch' or 'VisionDocumentSearch'

        Returns:
            Dict: Search results and generated response
        """
        data = {"query": query}

        if model:
            data["model"] = model
        if workspace_ids:
            data["workspace_ids"] = workspace_ids
        if file_ids:
            data["file_ids"] = file_ids
        if chat_session_id:
            data["chat_session_id"] = chat_session_id
        if company_scope is not None:
            data["company_scope"] = company_scope
        if private_scope is not None:
            data["private_scope"] = private_scope
        if tool:
            data["tool"] = tool

        return self._make_request("POST", "/api/v2/chat/document-search", data)


class AsyncDocumentSearch:
    """Async implementation for document search API endpoint with concurrent processing capabilities."""

    def __init__(self, api_key: str, base_url: str, default_model: str = None):
        """
        Initialize the async API client with authentication, base URL and default model.

        Args:
            api_key (str): The API key for authentication
            base_url (str): The base URL for the API endpoints
            default_model (str, optional): Default model to use for requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self.default_model = default_model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _make_async_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        data: Dict = None,
    ) -> Dict:
        """
        Make an async HTTP request to the Paradigm API.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint path
            data (Dict, optional): Request payload

        Returns:
            Dict: API response
        """
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "POST":
                async with session.post(url, headers=self.headers, json=data, ssl=False) as response:
                    response.raise_for_status()
                    if response.status == 204:  # No Content
                        return {}
                    return await response.json()
            elif method.upper() == "GET":
                async with session.get(url, headers=self.headers, ssl=False) as response:
                    response.raise_for_status()
                    if response.status == 204:  # No Content
                        return {}
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Request processing failed: {str(e)}")

    def _ensure_model(self, model: str = None) -> str:
        """
        Ensure a model is specified, using the default if none is provided.

        Args:
            model (str, optional): Model name to use

        Returns:
            str: The model name to use

        Raises:
            ValueError: If no model is specified and no default is set
        """
        model = model or self.default_model
        if not model:
            raise ValueError("No model specified and no default model set")
        return model

    async def execute_single(
        self,
        session: aiohttp.ClientSession,
        query: str,
        model: str = None,
        workspace_ids: List[int] = None,
        file_ids: List[int] = None,
        chat_session_id: int = None,
        company_scope: bool = None,
        private_scope: bool = None,
        tool: str = "DocumentSearch",
    ) -> Dict:
        """
        Execute a single document search request asynchronously.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use
            query (str): The search query
            model (str, optional): Model to use, defaults to the instance default model
            workspace_ids (List[int], optional): List of workspace IDs to search
            file_ids (List[int], optional): List of document IDs to search
            chat_session_id (int, optional): Chat session ID for follow-up
            company_scope (bool, optional): Include documents from company collection
            private_scope (bool, optional): Include documents from private collection
            tool (str, optional): Tool to use, either 'DocumentSearch' or 'VisionDocumentSearch'

        Returns:
            Dict: Search results and generated response
        """
        data = {"query": query}

        if model:
            data["model"] = model
        if workspace_ids:
            data["workspace_ids"] = workspace_ids
        if file_ids:
            data["file_ids"] = file_ids
        if chat_session_id:
            data["chat_session_id"] = chat_session_id
        if company_scope is not None:
            data["company_scope"] = company_scope
        if private_scope is not None:
            data["private_scope"] = private_scope
        if tool:
            data["tool"] = tool

        return await self._make_async_request(session, "POST", "/api/v2/chat/document-search", data)

    async def execute_batch(
        self,
        requests_data: List[Dict],
        max_concurrent: int = 10,
        progress_callback=None
    ) -> List[Dict]:
        """
        Execute multiple document search requests concurrently with controlled parallelism.

        Args:
            requests_data (List[Dict]): List of request dictionaries, each containing parameters for execute_single
            max_concurrent (int): Maximum number of concurrent requests (default: 10)
            progress_callback (callable, optional): Callback function to report progress

        Returns:
            List[Dict]: List of results corresponding to each request
        """
        if not requests_data:
            return []

        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(session, request_data, index):
            async with semaphore:
                try:
                    result = await self.execute_single(session, **request_data)
                    if progress_callback:
                        progress_callback(index + 1, len(requests_data))
                    return {"index": index, "result": result, "error": None}
                except Exception as e:
                    if progress_callback:
                        progress_callback(index + 1, len(requests_data))
                    return {"index": index, "result": None, "error": str(e)}

        # Create aiohttp session with connection limits
        connector = aiohttp.TCPConnector(limit=max_concurrent * 2, ssl=False)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout per request

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all requests
            tasks = [
                execute_with_semaphore(session, request_data, i)
                for i, request_data in enumerate(requests_data)
            ]

            # Execute all tasks concurrently
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Sort results by original index to maintain order
            results = [None] * len(requests_data)
            for completed_result in completed_results:
                if isinstance(completed_result, Exception):
                    # Handle exceptions that weren't caught in execute_with_semaphore
                    continue
                index = completed_result["index"]
                results[index] = {
                    "result": completed_result["result"],
                    "error": completed_result["error"]
                }

            return results


class DocumentSearchAlt(ParadigmAPI):
    """Alternative implementation for document search that breaks down the process into multiple steps.
    Used when document-search endpoint is not working."""

    def _query_documents(self, query: str, n: int = 5) -> List[Dict]:
        """
        Query documents using the query endpoint to get top N chunks.

        Args:
            query (str): The search query
            n (int, optional): Number of top chunks to retrieve

        Returns:
            List[Dict]: List of retrieved chunks with their metadata
        """
        data = {"query": query, "n": n}

        return self._make_request("POST", "/api/v2/query", data)

    def _generate_response(
        self, query: str, context_chunks: List[Dict], model: str = None
    ) -> tuple[Dict, List[Dict]]:
        """
        Generate a response using chat completions based on retrieved chunks.
        Also returns the messages list used as the prompt.

        Args:
            query (str): The original query
            context_chunks (List[Dict]): List of relevant text chunks
            model (str, optional): Model to use for completion

        Returns:
            tuple[Dict, List[Dict]]: A tuple containing the Chat completion response and the messages list sent to the API.
        """
        # Format context from chunks with explicit markers
        context_parts = []
        chunk_counter = 1
        for chunk_set in context_chunks:
            # Iterate through the actual chunks list within the chunk_set
            for chunk in chunk_set.get("chunks", []):
                if "text" in chunk:
                    source = chunk.get("metadata", {}).get("source", "N/A")
                    context_parts.append(
                        f"--- Start Chunk {chunk_counter} (Source: {source}) ---"
                    )
                    context_parts.append(chunk["text"])
                    context_parts.append(
                        f"--- End Chunk {chunk_counter} ---\n"
                    )  # Add extra newline for separation
                    chunk_counter += 1

        context = "\n".join(context_parts)

        # Create chat completion messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the user's question. "
                "Only use information from the provided context. If you cannot answer based on the context, "
                "say so.",
            },
            {
                "role": "user",
                "content": f"QUESTION OF USER : {query}\n\nCONTEXT TO USE : {context}",
            },
        ]

        # Use ChatCompletions to generate response
        chat = ChatCompletions(self.api_key, self.base_url, self.default_model)
        response = chat.execute(messages, model)
        return response, messages  # Return both the response and the prompt messages

    def execute(
        self,
        query: str,
        model: str = None,
        workspace_ids: List[int] = None,
        company_scope: bool = None,
        private_scope: bool = None,
    ) -> Dict:
        """
        Execute the alternative document search process.

        Args:
            query (str): The search query
            model (str, optional): Model to use, defaults to the instance default model
            workspace_ids (List[int], optional): List of workspace IDs to search
            company_scope (bool, optional): Include documents from company collection
            private_scope (bool, optional): Include documents from private collection

        Returns:
            Dict: Generated response with context and the prompt used.
        """
        # Step 1: Query documents to get relevant chunks
        chunks = self._query_documents(query)

        if not chunks:
            return {"error": "No relevant content found via query endpoint."}

        # Step 2: Generate response using retrieved chunks
        response, prompt_messages = self._generate_response(query, chunks, model)

        # Return combined result
        return {
            "response": response,
            "source_chunks": chunks,
            "prompt_messages": prompt_messages,
        }


class Models(ParadigmAPI):
    """Implementation for models API endpoint."""

    def execute(self) -> List[str]:
        """
        Fetch and extract technical names from the LiveIntelligence API response.

        Returns:
            List[str]: List of technical names of available models.

        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the API response structure is invalid or JSON parsing fails.
        """
        response = self._make_request("GET", "/api/v2/models")

        # Simple check of expected structure
        if (
            not isinstance(response, dict)
            or "data" not in response
            or not isinstance(response["data"], list)
        ):
            raise ValueError("Invalid response structure received from API")

        technical_names = [
            model["technical_name"]
            for model in response["data"]
            if isinstance(model, dict) and "technical_name" in model
        ]

        return technical_names


class Files(ParadigmAPI):
    """Implementation for files API endpoint."""

    def execute(
        self,
        company_scope: Optional[bool] = None,
        private_scope: Optional[bool] = None,
        workspace_scope: Optional[int] = None,
        page: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieves files IDs + filenames from the LiveIntelligence API based on the specified scope.

        Args:
            company_scope (bool, optional): Include documents from company collection
            private_scope (bool, optional): Include documents from user's private collection
            workspace_scope (int, optional): Include documents from specific workspace ID
            page (int, optional): Page number for pagination

        Returns:
            List[Dict[str, str]]: List of dictionaries, each containing 'id' and 'filename' for a file.

        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the API response structure is invalid or JSON parsing fails.
        """
        params = {}

        # Add optional parameters if they are not None
        if company_scope is not None:
            params["company_scope"] = company_scope
        if private_scope is not None:
            params["private_scope"] = private_scope
        if workspace_scope is not None:
            params["workspace_scope"] = workspace_scope
        if page is not None:
            params["page"] = page

        response = self._make_request("GET", "/api/v2/files", params=params)

        # Extract only 'id' and 'filename' for each file
        filtered_files = [
            {"id": file_obj["id"], "filename": file_obj["filename"]}
            for file_obj in response.get("data", [])
            if "id" in file_obj and "filename" in file_obj
        ]

        return filtered_files


class FileUploader(ParadigmAPI):
    """Implementation for file uploading operations."""

    def open_session(self, ingestion_pipeline: Optional[str] = None) -> str:
        """
        Open a new upload session.

        Args:
            ingestion_pipeline (Optional[str]): The ingestion pipeline version to use (e.g., "v2.2.1").

        Returns:
            str: The UUID of the opened upload session.
        """
        payload = {}
        if ingestion_pipeline:
            payload["ingestion_pipeline"] = ingestion_pipeline

        response = self._make_request(
            "POST", "/api/v2/upload-session", data=payload if payload else None
        )

        session_uuid = response.get("uuid")
        if not session_uuid:
            raise ValueError(
                f"Failed to open upload session or UUID not found in response. Response: {response}"
            )
        return session_uuid

    def upload_file_to_session(
        self,
        session_uuid: str,
        file_path: str,
        title: Optional[str] = None,
        metadata_filename: Optional[str] = None,
        collection_type: Optional[str] = None,
        workspace_id: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[float] = None,
        ocr_agent: Optional[str] = None,
        ocr_complete_doc_table_extraction: Optional[bool] = None,
        ocr_hi_res_model_name: Optional[str] = None,
        ocr_strategy: Optional[str] = None,
        ocr_timeout: Optional[int] = None,
        ocr_url: Optional[str] = None,
        is_visual_ingestion_on: Optional[bool] = None,
        default_pipeline: Optional[str] = None,
    ) -> Dict:
        """
        Upload a file to an existing upload session.

        Args:
            session_uuid (str): UUID of the upload session.
            file_path (str): Path to the local file to upload.
            title (Optional[str]): Title for the document.
            metadata_filename (Optional[str]): Filename for API metadata. Defaults to file_path's basename.
            collection_type (Optional[str]): Collection type ("private", "company", "workspace").
            workspace_id (Optional[int]): Workspace ID (required if collection_type is "workspace").
            chunk_size (Optional[int]): Custom chunk size.
            chunk_overlap (Optional[float]): Custom chunk overlap ratio (0.0 to 1.0).
            ocr_agent (Optional[str]): OCR agent.
            ocr_complete_doc_table_extraction (Optional[bool]): OCR table extraction flag.
            ocr_hi_res_model_name (Optional[str]): High-resolution OCR model.
            ocr_strategy (Optional[str]): OCR strategy.
            ocr_timeout (Optional[int]): OCR timeout in seconds.
            ocr_url (Optional[str]): OCR service URL.
            is_visual_ingestion_on (Optional[bool]): Visual ingestion flag.
            default_pipeline (Optional[str]): Default ingestion pipeline.

        Returns:
            Dict: API response from the file upload.
        """

        form_data_fields = {}
        if title:
            form_data_fields["title"] = title

        actual_upload_filename = os.path.basename(file_path)
        form_data_fields["filename"] = (
            metadata_filename
            if metadata_filename is not None
            else actual_upload_filename
        )

        if collection_type:
            form_data_fields["collection_type"] = collection_type
            if collection_type.lower() == "workspace" and workspace_id is None:
                raise ValueError(
                    "workspace_id is required when collection_type is 'workspace'."
                )

        if workspace_id is not None:
            form_data_fields["workspace_id"] = workspace_id

        # Optional processing parameters
        if chunk_size is not None:
            form_data_fields["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            form_data_fields["chunk_overlap"] = chunk_overlap

        # OCR parameters (uppercase as per API schema)
        if ocr_agent:
            form_data_fields["OCR_AGENT"] = ocr_agent
        if ocr_complete_doc_table_extraction is not None:
            form_data_fields["OCR_COMPLETE_DOC_TABLE_EXTRACTION"] = (
                ocr_complete_doc_table_extraction
            )
        if ocr_hi_res_model_name:
            form_data_fields["OCR_HI_RES_MODEL_NAME"] = ocr_hi_res_model_name
        if ocr_strategy:
            form_data_fields["OCR_STRATEGY"] = ocr_strategy
        if ocr_timeout is not None:
            form_data_fields["OCR_TIMEOUT"] = ocr_timeout
        if ocr_url:
            form_data_fields["OCR_URL"] = ocr_url
        if is_visual_ingestion_on is not None:
            form_data_fields["IS_VISUAL_INGESTION_ON"] = is_visual_ingestion_on
        if default_pipeline:
            form_data_fields["DEFAULT_PIPELINE"] = default_pipeline

        try:
            with open(file_path, "rb") as f_stream:
                files_payload = {"file": (actual_upload_filename, f_stream)}
                endpoint = f"/api/v2/upload-session/{session_uuid}"
                return self._make_request(
                    "POST", endpoint, data=form_data_fields, files=files_payload
                )
        except FileNotFoundError:
            raise ValueError(f"File not found at path: {file_path}")
        except requests.exceptions.HTTPError as he:
            raise
        except Exception as e:
            endpoint_for_error = (
                f"/api/v2/upload-session/{session_uuid}"
                if "session_uuid" in locals()
                else "unknown_endpoint"
            )
            raise RuntimeError(
                f"An unexpected error ({type(e).__name__}: {e}) occurred while preparing to upload file "
                f"{actual_upload_filename} to {endpoint_for_error}."
            ) from e

    def get_session_details(self, session_uuid: str) -> Dict:
        """
        Get details of an upload session.

        Args:
            session_uuid (str): The UUID of the upload session.

        Returns:
            Dict: Details of the upload session.
        """
        endpoint = f"/api/v2/upload-session/{session_uuid}"
        return self._make_request("GET", endpoint)

    def delete_session(self, session_uuid: str) -> Dict:
        """
        Delete an upload session and its documents.

        Args:
            session_uuid (str): The UUID of the upload session.

        Returns:
            Dict: Empty dict on success (API returns 204 No Content).
        """
        endpoint = f"/api/v2/upload-session/{session_uuid}"
        return self._make_request("DELETE", endpoint)

    def deactivate_all_sessions(self) -> Dict:
        """
        Deactivate the last user sessions.
        Corresponds to POST /api/v2/upload-session/deactivate.

        Returns:
            Dict: API response.
        """
        return self._make_request("POST", "/api/v2/upload-session/deactivate")

    def execute(self, session_uuid: str, file_path: str, **kwargs) -> Dict:
        """
        Main execution method, uploads a file to an existing session.
        Alias for upload_file_to_session.

        Args:
            session_uuid (str): UUID of the upload session.
            file_path (str): Path to the local file to upload.
            **kwargs: Additional parameters for upload_file_to_session.

        Returns:
            Dict: API response from the file upload.
        """
        return self.upload_file_to_session(session_uuid, file_path, **kwargs)

    def _wait_for_session_completion(self, session_uuid: str, max_wait_time: int = 3600, check_interval: int = 30) -> bool:
        """
        Wait for all documents in a session to complete processing.

        Args:
            session_uuid (str): UUID of the upload session to monitor.
            max_wait_time (int): Maximum time to wait in seconds (default: 1 hour).
            check_interval (int): Time between status checks in seconds (default: 30 seconds).

        Returns:
            bool: True if all documents completed processing, False if timeout occurred.
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                session_details = self.get_session_details(session_uuid)
                documents = session_details.get("documents", [])

                if not documents:
                    # No documents in session, consider it complete
                    return True

                # Check if all documents have completed processing
                pending_documents = [doc for doc in documents if doc.get("status") == "pending"]

                if not pending_documents:
                    # All documents have completed processing
                    print(f"All {len(documents)} documents in session {session_uuid} have completed processing.")
                    return True

                print(f"Waiting for {len(pending_documents)} documents to complete processing in session {session_uuid}...")
                time.sleep(check_interval)

            except Exception as e:
                print(f"Error checking session status: {e}")
                time.sleep(check_interval)

        print(f"Timeout waiting for session {session_uuid} to complete after {max_wait_time} seconds.")
        return False

    def upload_files_in_batches(
        self,
        file_paths: List[str],
        batch_size: int = 30,
        max_wait_time: int = 3600,
        check_interval: int = 30,
        ingestion_pipeline: Optional[str] = None,
        **upload_kwargs
    ) -> Dict:
        """
        Upload the files to our personal space with robust batch processing.

        This method implements a respectful upload system that:
        1. Limits uploads to a maximum of 30 files per batch to prevent system overload
        2. Monitors session status between batches using get_session_details()
        3. Waits for all documents to complete processing before proceeding
        4. Properly cleans up sessions and creates new ones for each batch
        5. Continues iteratively until all files are uploaded and processed

        Args:
            file_paths (List[str]): List of file paths to upload.
            batch_size (int): Maximum number of files per batch (default: 10, max: 10).
            max_wait_time (int): Maximum time to wait for each batch to complete (default: 1 hour).
            check_interval (int): Time between status checks in seconds (default: 30 seconds).
            ingestion_pipeline (Optional[str]): The ingestion pipeline version to use.
            **upload_kwargs: Additional parameters to pass to upload_file_to_session.

        Returns:
            Dict: Summary of the upload process including success/failure counts and details.
        """
        # Ensure batch size doesn't exceed the maximum limit
        batch_size = min(batch_size, 30)

        if not file_paths:
            return {
                "status": "completed",
                "total_files": 0,
                "successful_uploads": 0,
                "failed_uploads": 0,
                "batches_processed": 0,
                "details": []
            }

        total_files = len(file_paths)
        successful_uploads = 0
        failed_uploads = 0
        batches_processed = 0
        upload_details = []

        print(f"Starting batch upload of {total_files} files with batch size {batch_size}")

        # Process files in batches
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            current_batch = file_paths[batch_start:batch_end]
            batch_number = batches_processed + 1

            print(f"\n--- Processing Batch {batch_number} ({len(current_batch)} files) ---")

            batch_details = {
                "batch_number": batch_number,
                "files": current_batch,
                "session_uuid": None,
                "successful_files": [],
                "failed_files": [],
                "status": "started"
            }

            try:
                # Step 1: Create a new upload session for this batch
                print(f"Creating new upload session for batch {batch_number}...")
                session_uuid = self.open_session(ingestion_pipeline=ingestion_pipeline)
                batch_details["session_uuid"] = session_uuid
                print(f"Created session: {session_uuid}")

                # Step 2: Upload all files in the current batch
                print(f"Uploading {len(current_batch)} files to session {session_uuid}...")
                for file_path in current_batch:
                    try:
                        print(f"  Uploading: {os.path.basename(file_path)}")
                        response = self.upload_file_to_session(
                            session_uuid=session_uuid,
                            file_path=file_path,
                            **upload_kwargs
                        )
                        batch_details["successful_files"].append({
                            "file_path": file_path,
                            "response": response
                        })
                        successful_uploads += 1
                        print(f"  ✓ Successfully uploaded: {os.path.basename(file_path)}")

                    except Exception as e:
                        error_msg = f"Failed to upload {file_path}: {str(e)}"
                        print(f"  ✗ {error_msg}")
                        batch_details["failed_files"].append({
                            "file_path": file_path,
                            "error": error_msg
                        })
                        failed_uploads += 1

                # Step 3: Wait for all documents in this batch to complete processing
                if batch_details["successful_files"]:
                    print(f"Waiting for batch {batch_number} to complete processing...")
                    completion_success = self._wait_for_session_completion(
                        session_uuid=session_uuid,
                        max_wait_time=max_wait_time,
                        check_interval=check_interval
                    )

                    if completion_success:
                        print(f"✓ Batch {batch_number} completed successfully")
                        batch_details["status"] = "completed"
                    else:
                        print(f"⚠ Batch {batch_number} timed out waiting for completion")
                        batch_details["status"] = "timeout"
                else:
                    print(f"No files were successfully uploaded in batch {batch_number}")
                    batch_details["status"] = "no_uploads"

                # Step 4: Clean up the current session
                try:
                    print(f"Closing session {session_uuid} in 60 seconds ...")
                    time.sleep(60)
                    self.deactivate_all_sessions()
                    print(f"✓ Session {session_uuid} closed successfully")
                except Exception as e:
                    print(f"⚠ Warning: Failed to close session {session_uuid}: {e}")

            except Exception as e:
                error_msg = f"Batch {batch_number} failed: {str(e)}"
                print(f"✗ {error_msg}")
                batch_details["status"] = "failed"
                batch_details["error"] = error_msg
                # Mark all files in this batch as failed if session creation failed
                for file_path in current_batch:
                    if file_path not in [f["file_path"] for f in batch_details["failed_files"]]:
                        batch_details["failed_files"].append({
                            "file_path": file_path,
                            "error": f"Batch session creation failed: {str(e)}"
                        })
                        failed_uploads += 1

            upload_details.append(batch_details)
            batches_processed += 1

            # Brief pause between batches to be respectful to the system
            if batch_end < total_files:
                print(f"Pausing briefly before next batch...")
                time.sleep(2)

        # Final summary
        print(f"\n=== Upload Summary ===")
        print(f"Total files: {total_files}")
        print(f"Successful uploads: {successful_uploads}")
        print(f"Failed uploads: {failed_uploads}")
        print(f"Batches processed: {batches_processed}")
        print(f"Success rate: {(successful_uploads/total_files)*100:.1f}%" if total_files > 0 else "N/A")

        return {
            "status": "completed",
            "total_files": total_files,
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "batches_processed": batches_processed,
            "success_rate": (successful_uploads/total_files)*100 if total_files > 0 else 0,
            "details": upload_details
        }

    def upload_files_to_personal_space(
        self,
        file_paths: List[str],
        collection_type: str = "private",
        workspace_id: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Upload the files to our personal space using robust batch processing.

        This is a convenience method that calls upload_files_in_batches with
        appropriate settings for uploading to personal/private space.

        Args:
            file_paths (List[str]): List of file paths to upload.
            collection_type (str): Collection type ("private", "company", "workspace").
            workspace_id (Optional[int]): Workspace ID (required if collection_type is "workspace").
            **kwargs: Additional parameters to pass to the upload process.

        Returns:
            Dict: Summary of the upload process.
        """
        # Set up upload parameters for personal space
        upload_params = {
            "collection_type": collection_type,
            **kwargs
        }

        if collection_type == "workspace" and workspace_id is not None:
            upload_params["workspace_id"] = workspace_id
        elif collection_type == "workspace" and workspace_id is None:
            raise ValueError("workspace_id is required when collection_type is 'workspace'")

        return self.upload_files_in_batches(
            file_paths=file_paths,
            **upload_params
        )


if __name__ == "__main__":
    # Print models
    API_KEY = os.getenv("PROD_KEY")
    BASE_URL = os.getenv("PROD_URL")

    documents_api = DocumentSearch(api_key=API_KEY, base_url=BASE_URL)
    response = documents_api.execute(query="which are the top 2 ancient grains among those who have tried them", workspace_ids=[19], company_scope=False, private_scope=False, tool="VisionDocumentSearch")
    print(response)