"""Encrypt OAuth tokens at rest. The DB never holds plaintext tokens."""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from ..config import settings


class ConnectionsCryptoError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = settings.connections_enc_key
    if not key:
        raise ConnectionsCryptoError(
            "CONNECTIONS_ENC_KEY is not set — cannot encrypt/decrypt account tokens. "
            'Generate one: python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        )
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        raise ConnectionsCryptoError(f"CONNECTIONS_ENC_KEY is invalid: {exc}") from exc


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ConnectionsCryptoError("Token decryption failed (wrong key?)") from exc
