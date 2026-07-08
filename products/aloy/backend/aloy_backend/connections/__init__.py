"""Native account connections (OAuth) — Aloy owns the tokens and the data path.

The connect-engine (this package) is provider-agnostic: a ``ProviderSpec``
describes an OAuth provider, and the store/crypto handle the token lifecycle.
Adding an app = a new ProviderSpec + its tools. Gmail is the first.
"""

from .providers import PROVIDERS, ProviderSpec, available_providers, get_provider

__all__ = ["PROVIDERS", "ProviderSpec", "available_providers", "get_provider"]
