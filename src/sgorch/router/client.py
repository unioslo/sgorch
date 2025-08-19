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
            endpoint = self.config.endpoints.add
            
            # Try different payload formats that routers commonly accept
            payloads_to_try = [
                {"url": worker_url},
                {"worker": worker_url},
                {"endpoint": worker_url},
                {"worker_url": worker_url},
                worker_url,  # Some routers accept plain strings
            ]
            
            last_error = None
            
            for payload in payloads_to_try:
                try:
                    if isinstance(payload, str):
                        # For string payloads, try both JSON and form data
                        response = self.client.post(endpoint, json=payload)
                        if response.status_code >= 400:
                            response = self.client.post(endpoint, data={"url": payload})
                    else:
                        response = self.client.post(endpoint, json=payload)
                    
                    response.raise_for_status()
                    
                    # Success
                    logger.info(f"Successfully added worker: {worker_url}")
                    return
                    
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 400:
                        # Bad request, try next payload format
                        continue
                    else:
                        # Other HTTP errors, re-raise immediately
                        raise
            
            # If we get here, all payload formats failed
            if last_error:
                raise last_error
                
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
            endpoint = self.config.endpoints.remove
            
            # Try different approaches for removing workers
            # Method 1: DELETE with URL in path
            try:
                # Some routers expect URL-encoded worker URL in path
                import urllib.parse
                encoded_url = urllib.parse.quote(worker_url, safe='')
                delete_endpoint = f"{endpoint}/{encoded_url}"
                response = self.client.delete(delete_endpoint)
                
                if response.status_code not in [404, 405]:  # Not method not allowed or not found
                    response.raise_for_status()
                    logger.info(f"Successfully removed worker: {worker_url}")
                    return
            except httpx.HTTPStatusError:
                pass  # Try next method
            
            # Method 2: DELETE with JSON body
            try:
                response = self.client.delete(endpoint, json={"url": worker_url})
                if response.status_code not in [404, 405]:
                    response.raise_for_status()
                    logger.info(f"Successfully removed worker: {worker_url}")
                    return
            except httpx.HTTPStatusError:
                pass  # Try next method
            
            # Method 3: POST to remove endpoint
            try:
                response = self.client.post(endpoint, json={"url": worker_url})
                response.raise_for_status()
                logger.info(f"Successfully removed worker: {worker_url}")
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Worker not found is OK
                    logger.info(f"Worker not found in router (already removed): {worker_url}")
                    return
                raise
                
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