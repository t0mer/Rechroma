"""Optional shared-token auth for the API (CLAUDE.md §6).

If ``web_auth_token`` is configured, every ``/api/v1`` request must present it as
``Authorization: Bearer <token>`` or ``X-API-Token: <token>``. If it is unset the
API is open (a prominent warning is logged at startup) — intended to sit behind a
reverse proxy.
"""

import hmac

from fastapi import Header, HTTPException, Request, status


def verify_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> None:
    """FastAPI dependency enforcing the shared token when one is configured."""
    expected: str | None = request.app.state.settings.web_auth_token
    if not expected:
        return  # open mode
    provided = x_api_token
    if provided is None and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:]
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
