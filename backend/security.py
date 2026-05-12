"""HMAC token generation and verification for rating links."""

import hmac
import hashlib


def generate_rating_token(item_id: int, rating: str, secret: str) -> str:
    """HMAC-SHA256 over f"{item_id}:{rating}" using secret. Return first 32 hex chars."""
    msg = f"{item_id}:{rating}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return sig[:32]


def verify_rating_token(item_id: int, rating: str, token: str, secret: str) -> bool:
    """Recompute expected token. Compare with hmac.compare_digest (timing-safe).

    Return True if match, False otherwise. Never raise.
    """
    try:
        expected = generate_rating_token(item_id, rating, secret)
        return hmac.compare_digest(expected, token)
    except Exception:
        return False
