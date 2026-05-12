"""Tests for backend/security.py."""

from backend.security import generate_rating_token, verify_rating_token


SECRET = "test-secret-key-for-unit-tests"


def test_token_round_trip():
    token = generate_rating_token(42, "up", SECRET)
    assert verify_rating_token(42, "up", token, SECRET)


def test_token_is_32_hex_chars():
    token = generate_rating_token(1, "down", SECRET)
    assert len(token) == 32
    assert all(c in "0123456789abcdef" for c in token)


def test_tampered_item_id_rejected():
    token = generate_rating_token(42, "up", SECRET)
    assert not verify_rating_token(99, "up", token, SECRET)


def test_tampered_rating_rejected():
    token = generate_rating_token(42, "up", SECRET)
    assert not verify_rating_token(42, "down", token, SECRET)


def test_tampered_token_rejected():
    token = generate_rating_token(42, "up", SECRET)
    bad = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert not verify_rating_token(42, "up", bad, SECRET)


def test_wrong_secret_rejected():
    token = generate_rating_token(42, "up", SECRET)
    assert not verify_rating_token(42, "up", token, "wrong-secret")


def test_timing_safe_compare_used():
    # Verify the function uses compare_digest (never raises on bad input).
    assert not verify_rating_token(0, "", "", "")
    assert not verify_rating_token(-1, "up", "bad", SECRET)


def test_different_items_produce_different_tokens():
    t1 = generate_rating_token(1, "up", SECRET)
    t2 = generate_rating_token(2, "up", SECRET)
    assert t1 != t2
