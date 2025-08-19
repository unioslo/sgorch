import json
from typing import Optional, Set

import httpx

from ..logging_setup import get_logger
from ..config import RouterConfig, AuthConfig


logger = get_logger(__name__)


class RouterClient:
    """Client for interacting with SGLang router to manage workers."""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        
        # Set up HTTP client
        headers = {"Content-Type": "application/json"}
        
        # Add authentication if configured
        if config.auth:
            self._add_auth_headers(headers, config.auth)
        
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0
        )
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def _add_auth_headers(self, headers: dict, auth_config: AuthConfig) -> None:
        """Add authentication headers based on auth configuration."""
        if auth_config.type == "header":
            import os
            token_value = os.getenv(auth_config.header_value_env)
            if token_value:
                headers[auth_config.header_name] = token_value
            else:
                logger.warning(
                    f"Auth token environment variable {auth_config.header_value_env} not set"
                )
    
    def list(self) -> Set[str]:
        """List current workers registered with the router."""
        logger.debug("Listing workers from router")
        
        try:
            endpoint = self.config.endpoints.list
            response = self.client.get(endpoint)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                workers = set(data)
            elif isinstance(data, dict):
                # Common patterns: {"workers": [...]} or {"urls": [...]}
                workers = set(
                    data.get("workers", []) or 
                    data.get("urls", []) or
                    data.get("endpoints", [])
                )
            else:
                logger.warning(f"Unexpected response format from router list: {data}")
                workers = set()
            
            logger.debug(f"Router has {len(workers)} workers: {workers}")
            return workers
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error listing workers: {e.response.status_code} {e.response.text}")
            raise RouterError(f"Failed to list workers: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error listing workers: {e}")
            raise RouterError(f"Failed to list workers: {e}")
    
    def add(self, worker_url: str) -> None:
        """Add a worker to the router."""
        logger.info(f"Adding worker to router: {worker_url}")
        
        try:
            # SGLang router format: POST /add_worker?url=<worker_url>
            response = self.client.post(self.config.endpoints.add, params={"url": worker_url})
            response.raise_for_status()
            logger.info(f"Successfully added worker: {worker_url}")
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error adding worker: {e.response.status_code} {e.response.text}"
            logger.error(error_msg)
            raise RouterError(error_msg)
        except Exception as e:
            logger.error(f"Error adding worker: {e}")
            raise RouterError(f"Failed to add worker: {e}")
    
    def remove(self, worker_url: str) -> None:
        """Remove a worker from the router."""
        logger.info(f"Removing worker from router: {worker_url}")
        
        try:
            # SGLang router format: POST /remove_worker?url=<worker_url>
            response = self.client.post(self.config.endpoints.remove, params={"url": worker_url})
            
            if response.status_code == 404:
                logger.info(f"Worker not found in router (already removed): {worker_url}")
                return
                
            response.raise_for_status()
            logger.info(f"Successfully removed worker: {worker_url}")
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"Worker not found in router (already removed): {worker_url}")
                return
            error_msg = f"HTTP error removing worker: {e.response.status_code} {e.response.text}"
            logger.error(error_msg)
            raise RouterError(error_msg)
        except Exception as e:
            logger.error(f"Error removing worker: {e}")
            raise RouterError(f"Failed to remove worker: {e}")
    
    def health_check(self) -> bool:
        """Check if router is healthy and reachable."""
        logger.debug("Checking router health")
        
        try:
            # Try to list workers as a health check
            self.list()
            return True
        except Exception as e:
            logger.warning(f"Router health check failed: {e}")
            return False


class RouterError(Exception):
    """Exception raised for router operation errors."""
    pass