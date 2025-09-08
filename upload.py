import requests
import os
from pathlib import Path
from typing import Optional


class MammothConnect:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        workspace_id: Optional[int] = None,
        project_id: Optional[int] = None,
    ):
        """
        Initialize MammothConnect client.

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            base_url: Base URL for the API (optional)
            workspace_id: Workspace ID for constructing upload URLs (optional)
            project_id: Project ID for constructing upload URLs (optional)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or "https://higgs.mammoth.io"
        self.workspace_id = workspace_id
        self.project_id = project_id

    def upload_file(
        self, file_path: str, url: Optional[str] = None
    ) -> requests.Response:
        """
        Upload a file to the specified URL or constructed URL.

        Args:
            file_path: Path to the file to upload
            url: Optional URL to upload to. If not provided, constructs from workspace/project IDs

        Returns:
            requests.Response object

        Raises:
            ValueError: If no URL is provided and workspace/project IDs are not set
            FileNotFoundError: If the specified file doesn't exist
        """
        # Prepare headers with API credentials
        headers = {
            "X-API-KEY": self.api_key,
            "X-API-SECRET": self.api_secret,
            "Accept": "application/json, text/plain, */*",
        }

        # Construct URL if not provided
        if url is None:
            if self.workspace_id is None or self.project_id is None:
                raise ValueError(
                    "Either provide URL or set workspace_id and project_id in constructor"
                )
            url = f"{self.base_url}/api/v2/workspaces/{self.workspace_id}/projects/{self.project_id}/files"

        # Validate file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine content type based on file extension
        content_type = self._get_content_type(file_path.suffix.lower())

        # Upload file
        with open(file_path, "rb") as file:
            files = {"files": (file_path.name, file, content_type)}

            response = requests.post(url, headers=headers, files=files)

        return response

    def _get_content_type(self, file_extension: str) -> str:
        """
        Get content type based on file extension.

        Args:
            file_extension: File extension (e.g., '.csv', '.json')

        Returns:
            Content type string
        """
        content_types = {
            ".csv": "text/csv",
            ".json": "application/json",
            ".txt": "text/plain",
            ".xml": "application/xml",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pdf": "application/pdf",
        }
        return content_types.get(file_extension, "application/octet-stream")


client = MammothConnect(
    api_key="wuFlxzjO4BdEuC20uzfIZZVD_GkKsDrinvSnijZgT0I",
    api_secret="1t6-99S8vJR-dsByv6DxJ7TfZrMxH2RMFjxexjgvsLo",
    workspace_id=4,
    project_id=3,
)


def upload_file_to_mammoth(file_path: str) -> str:
    """Upload a file to Mammoth and return the upload status."""
    response = client.upload_file(file_path)
    return f"Upload status: {response.status_code}"
