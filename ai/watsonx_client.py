"""IBM watsonx.ai Client Module for IBM Granite Models.

Implements robust authentication with IBM Cloud IAM, payload serialization,
caching of bearer tokens, error handling, and inference calls to IBM Granite.
"""

import os
import time
import logging
import requests
from typing import Optional, Dict, Any

from config import Config

logger = logging.getLogger(__name__)


class WatsonxConfigurationError(Exception):
    """Raised when IBM watsonx.ai credentials or project settings are missing or invalid."""
    pass


class WatsonxAPIError(Exception):
    """Raised when the IBM watsonx.ai API returns an error during inference."""
    pass


class WatsonxClient:
    """Client for interacting with IBM Granite models via IBM watsonx.ai."""

    def __init__(
        self,
        apikey: Optional[str] = None,
        project_id: Optional[str] = None,
        url: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.apikey = (apikey or Config.WATSONX_APIKEY or "").strip()
        self.project_id = (project_id or Config.WATSONX_PROJECT_ID or "").strip()
        self.url = (url or Config.WATSONX_URL or "https://us-south.ml.cloud.ibm.com").rstrip("/")
        self.model_id = (model_id or Config.IBM_GRANITE_MODEL_ID or "ibm/granite-13b-chat-v2").strip()

        # Token caching
        self._iam_token: Optional[str] = None
        self._token_expiry: float = 0.0

    def is_configured(self) -> bool:
        """Check if minimum required credentials exist."""
        return bool(self.apikey and self.project_id)

    def _get_iam_token(self) -> str:
        """Authenticate with IBM Cloud IAM and retrieve an OAuth2 Bearer token.
        Caches token until 5 minutes before expiry (standard 1-hour validity).
        """
        now = time.time()
        if self._iam_token and now < (self._token_expiry - 300):
            return self._iam_token

        if not self.apikey:
            raise WatsonxConfigurationError(
                "IBM watsonx.ai API Key is not set. Please set WATSONX_APIKEY in .env or the Settings page."
            )

        iam_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.apikey,
        }

        try:
            response = requests.post(iam_url, headers=headers, data=data, timeout=30)
            if response.status_code != 200:
                error_msg = f"IBM IAM Authentication failed ({response.status_code}): {response.text}"
                logger.error(error_msg)
                raise WatsonxAPIError(error_msg)

            token_data = response.json()
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)

            if not access_token:
                raise WatsonxAPIError("No access token received from IBM Cloud IAM.")

            self._iam_token = access_token
            self._token_expiry = now + expires_in
            logger.info("Successfully authenticated with IBM Cloud IAM.")
            return self._iam_token

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during IBM IAM authentication: {e}")
            raise WatsonxAPIError(f"Could not connect to IBM IAM service: {str(e)}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1500,
        temperature: float = 0.3,
        min_new_tokens: int = 1,
        repetition_penalty: float = 1.05,
        stop_sequences: Optional[list] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate text using IBM Granite on IBM watsonx.ai.

        Args:
            prompt: Text prompt to send to Granite.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0).
            min_new_tokens: Minimum tokens to generate.
            repetition_penalty: Repetition penalty factor.
            stop_sequences: Optional list of stop sequences.
            model_id: Model to use (defaults to configured Granite model).

        Returns:
            Dict containing generated_text, model_id, token_counts, and metadata.
        """
        if not self.is_configured():
            raise WatsonxConfigurationError(
                "IBM watsonx.ai is not configured. Please supply your WATSONX_APIKEY and WATSONX_PROJECT_ID."
            )

        token = self._get_iam_token()
        selected_model = model_id or self.model_id
        endpoint = f"{self.url}/ml/v1/text/generation?version=2023-05-29"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        parameters = {
            "decoding_method": "greedy" if temperature == 0.0 else "sample",
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0.0:
            parameters["temperature"] = temperature
        if stop_sequences:
            parameters["stop_sequences"] = stop_sequences

        payload = {
            "input": prompt,
            "parameters": parameters,
            "model_id": selected_model,
            "project_id": self.project_id,
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=90)

            if response.status_code == 401 or response.status_code == 403:
                # Token might have expired or key is invalid
                self._iam_token = None
                raise WatsonxAPIError(
                    f"IBM watsonx.ai authorization denied ({response.status_code}). Check your API Key and Project ID permissions."
                )
            elif response.status_code == 404:
                raise WatsonxAPIError(
                    f"IBM watsonx.ai resource not found ({response.status_code}). Check that your Project ID '{self.project_id}' exists in the selected region."
                )
            elif response.status_code == 429:
                raise WatsonxAPIError(
                    "IBM watsonx.ai rate limit exceeded. Please wait a moment before trying again."
                )
            elif response.status_code != 200:
                raise WatsonxAPIError(
                    f"IBM watsonx.ai generation failed ({response.status_code}): {response.text}"
                )

            data = response.json()
            results = data.get("results", [])
            if not results:
                raise WatsonxAPIError("IBM watsonx.ai returned an empty result list.")

            first_result = results[0]
            generated_text = first_result.get("generated_text", "")
            input_token_count = first_result.get("input_token_count", 0)
            generated_token_count = first_result.get("generated_token_count", 0)
            stop_reason = first_result.get("stop_reason", "unknown")

            return {
                "generated_text": generated_text.strip(),
                "model_id": selected_model,
                "input_token_count": input_token_count,
                "generated_token_count": generated_token_count,
                "stop_reason": stop_reason,
            }

        except requests.exceptions.Timeout:
            raise WatsonxAPIError(
                "Request to IBM watsonx.ai timed out after 90 seconds. Try reducing document chunk size or max tokens."
            )
        except requests.exceptions.RequestException as e:
            raise WatsonxAPIError(f"Network error connecting to IBM watsonx.ai: {str(e)}")

    def test_connection(self) -> Dict[str, Any]:
        """Test authentication and project access with a lightweight query."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "Missing API Key or Project ID",
                "message": "Both WATSONX_APIKEY and WATSONX_PROJECT_ID must be configured.",
            }

        try:
            # First verify IAM token
            self._get_iam_token()
            # Send a micro-prompt to IBM Granite to verify project binding
            res = self.generate(
                prompt="Respond with the word 'READY' if this test query succeeds.",
                max_new_tokens=10,
                temperature=0.0,
            )
            return {
                "success": True,
                "message": f"Successfully connected to IBM Granite ({self.model_id}) on IBM watsonx.ai!",
                "response": res.get("generated_text", ""),
                "model_id": self.model_id,
            }
        except Exception as e:
            logger.warning(f"IBM watsonx.ai connection test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Connection to IBM watsonx.ai failed: {str(e)}",
            }
