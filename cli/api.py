"""HTTP client for the Coral Bricks backend.

Thin wrapper over `requests` that injects `Authorization: Bearer <key>`
and raises a friendly error on non-2xx responses. Server URL + API key
come from Config; either can be overridden via env vars.
"""

from __future__ import annotations

from typing import Any

import requests

from .config import Config

DEFAULT_TIMEOUT = 30


class ApiError(Exception):
    def __init__(self, status: int, message: str, payload: Any = None):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
        self.payload = payload


class AuthError(ApiError):
    """API key missing or invalid."""


class Client:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._session = requests.Session()

    @property
    def server_url(self) -> str:
        return self._cfg.effective_server_url().rstrip("/")

    def _headers(self, require_auth: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "coralbricks-cli/0.1.0",
        }
        key = self._cfg.effective_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif require_auth:
            raise AuthError(
                401,
                "Not logged in. Run `coralbricks login` first.",
            )
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict | None = None,
        require_auth: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Any:
        url = f"{self.server_url}{path}"
        try:
            resp = self._session.request(
                method,
                url,
                headers=self._headers(require_auth),
                json=json,
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise ApiError(0, f"Network error contacting {self.server_url}: {e}") from e

        if resp.status_code == 401 or resp.status_code == 403:
            msg = _extract_message(resp) or "Unauthorized."
            raise AuthError(resp.status_code, msg)
        if not resp.ok:
            raise ApiError(resp.status_code, _extract_message(resp) or resp.reason, _safe_json(resp))
        if not resp.content:
            return None
        body = _safe_json(resp)
        # Backend convention: { success: bool, data?: ..., error?: str }.
        # Unwrap `data` so callers can ignore the envelope.
        if isinstance(body, dict) and "success" in body:
            if body.get("success") is False:
                raise ApiError(
                    resp.status_code,
                    body.get("error") or body.get("message") or "Request failed",
                    body,
                )
            return body.get("data")
        return body

    def get(self, path: str, **kw: Any) -> Any:
        return self._request("GET", path, **kw)

    def post(self, path: str, json: Any = None, **kw: Any) -> Any:
        return self._request("POST", path, json=json, **kw)

    def delete(self, path: str, **kw: Any) -> Any:
        return self._request("DELETE", path, **kw)


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _extract_message(resp: requests.Response) -> str | None:
    data = _safe_json(resp)
    if isinstance(data, dict):
        for key in ("message", "error", "detail"):
            v = data.get(key)
            if isinstance(v, str):
                return v
    return None
