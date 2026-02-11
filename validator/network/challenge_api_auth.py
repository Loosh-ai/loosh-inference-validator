"""
Hotkey-signature authentication for outbound requests to the Challenge API.

Validators sign each request with their sr25519 hotkey using the same scheme
as Fiber (nonce + hotkey + SHA-256 body hash).  This eliminates the need for
shared API keys — the validator's on-chain identity is the credential.

The Challenge API verifies the signature and checks that the hotkey belongs
to a known validator.

Usage
-----
    from validator.network.challenge_api_auth import get_auth_headers

    headers = get_auth_headers(body=json_bytes)
    # → {"X-Hotkey": "5G...", "X-Nonce": "...", "X-Signature": "0x..."}

    # Merge with your existing headers and send the request.

Backwards compatibility
-----------------------
If the validator keypair has not been set (``set_validator_keypair`` not yet
called), ``get_auth_headers`` returns an empty dict so callers can fall
through to legacy API-key authentication without crashing.
"""

import hashlib
import time
from typing import Any, Dict, Optional

from loguru import logger


def get_auth_headers(
    body: Optional[bytes] = None,
    *,
    keypair: Optional[Any] = None,
) -> Dict[str, str]:
    """Build hotkey-signature auth headers for a Challenge API request.

    Parameters
    ----------
    body : bytes | None
        Raw request body (JSON-encoded).  For GET requests pass ``None``.
    keypair : Keypair | None
        Explicit keypair override.  When ``None`` the module-level cached
        keypair from ``validator.network.fiber_client`` is used.

    Returns
    -------
    dict
        ``{"X-Hotkey": ..., "X-Nonce": ..., "X-Signature": ...}`` on
        success, or an **empty dict** if no keypair is available (allows
        callers to fall back to API-key auth).
    """
    if keypair is None:
        from validator.network.fiber_client import get_validator_keypair
        keypair = get_validator_keypair()

    if keypair is None:
        # Keypair not loaded yet (startup race) — caller should fall back
        return {}

    try:
        # Nonce: epoch timestamp with microsecond precision (matches
        # Fiber's generate_nonce pattern but also doubles as an age check
        # on the server side).
        nonce = str(time.time())

        hotkey_ss58: str = keypair.ss58_address

        # Build signing message — same scheme as Fiber:
        #   "{nonce}:{hotkey}:{sha256(body)}"  or  "{nonce}:{hotkey}"
        if body:
            body_hash = hashlib.sha256(body).hexdigest()
            message = f"{nonce}:{hotkey_ss58}:{body_hash}"
        else:
            message = f"{nonce}:{hotkey_ss58}"

        # sr25519 signature via substrateinterface Keypair
        sig_bytes = keypair.sign(message)
        signature = f"0x{sig_bytes.hex()}"

        return {
            "X-Hotkey": hotkey_ss58,
            "X-Nonce": nonce,
            "X-Signature": signature,
        }
    except Exception as e:
        logger.warning(f"Failed to generate hotkey auth headers: {e}")
        return {}


def merge_auth_headers(
    existing_headers: Dict[str, str],
    body: Optional[bytes] = None,
    *,
    api_key: Optional[str] = None,
    keypair: Optional[Any] = None,
) -> Dict[str, str]:
    """Return *existing_headers* enriched with authentication.

    Tries hotkey signature first.  If that is unavailable (no keypair) and
    an ``api_key`` is provided, falls back to ``X-API-Key``.

    This is the single call-site-friendly helper: just pass whatever you
    have and it picks the best available auth method.
    """
    auth = get_auth_headers(body=body, keypair=keypair)
    headers = dict(existing_headers)

    if auth:
        # Add hotkey signature headers
        headers.update(auth)

        # ALSO include the legacy API key if available so that an older
        # Challenge API (which doesn't understand hotkey auth yet) can
        # still authenticate this request via X-API-Key.  The new
        # Challenge API checks hotkey first and never reaches the API-key
        # path, so the extra header is harmless.
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    # No hotkey available — use legacy API key only
    if api_key:
        headers["X-API-Key"] = api_key
        return headers

    # No auth available — return headers as-is
    return headers
