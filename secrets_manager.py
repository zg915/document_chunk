"""
Google Secret Manager integration for secure configuration management.
"""

import os
import logging
from typing import Optional
from google.cloud import secretmanager

logger = logging.getLogger(__name__)

class SecretManager:
    """Handle Google Secret Manager operations."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Secret Manager client.
        
        Args:
            project_id: GCP Project ID (defaults to metadata service)
        """
        self.project_id = project_id or self._get_project_id()
        self.client = secretmanager.SecretManagerServiceClient()
    
    def _get_project_id(self) -> str:
        """Get project ID from metadata service or environment."""
        try:
            import requests
            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
                timeout=5
            )
            return response.text
        except Exception:
            # Fallback to environment variable
            return os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
    
    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """
        Retrieve a secret from Secret Manager.
        
        Args:
            secret_name: Name of the secret
            version: Secret version (default: latest)
            
        Returns:
            Secret value or None if not found
        """
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def get_weaviate_config(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get Weaviate configuration from Secret Manager.
        
        Returns:
            Tuple of (url, api_key)
        """
        url = self.get_secret("weaviate-url")
        api_key = self.get_secret("weaviate-api-key")
        return url, api_key
    
    def get_marker_config(self) -> Optional[str]:
        """
        Get Marker API key from Secret Manager.
        
        Returns:
            Marker API key
        """
        return self.get_secret("marker-api-key")

# Global instance
secret_manager = SecretManager()

def get_weaviate_config() -> tuple[Optional[str], Optional[str]]:
    """Get Weaviate configuration with fallback to environment variables."""
    # Try Secret Manager first
    url, api_key = secret_manager.get_weaviate_config()
    
    # Fallback to environment variables
    if not url:
        url = os.getenv("WEAVIATE_URL")
    if not api_key:
        api_key = os.getenv("WEAVIATE_API_KEY")
    
    return url, api_key

def get_marker_config() -> Optional[str]:
    """Get Marker configuration with fallback to environment variables."""
    # Try Secret Manager first
    api_key = secret_manager.get_marker_config()
    
    # Fallback to environment variables
    if not api_key:
        api_key = os.getenv("MARKER_API_KEY")
    
    return api_key
