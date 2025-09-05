"""
Clean configuration management for Pori agent system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MemoryConfig:
    """Clean memory configuration."""

    persistent: bool = False
    vector: bool = False
    vector_backend: str = "weaviate"

    # Vector store specific configs
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class: str = "AgentMemory"
    weaviate_vectorizer: str = "text2vec-huggingface"

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create configuration from environment variables."""
        return cls(
            persistent=False,
            vector=os.getenv("VECTOR", "false").lower() in {"1", "true", "yes", "on"},
            vector_backend=os.getenv("VECTOR_BACKEND", "weaviate"),
            weaviate_url=os.getenv("WEAVIATE_URL"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_class=os.getenv("WEAVIATE_CLASS", "AgentMemory"),
            weaviate_vectorizer=os.getenv(
                "WEAVIATE_VECTORIZER", "text2vec-huggingface"
            ),
        )

    def to_vector_config(self) -> Optional[Dict[str, Any]]:
        """Convert to vector store configuration."""
        if not self.vector:
            return None

        config = {"backend": self.vector_backend}

        if self.vector_backend == "weaviate":
            config.update(
                {
                    "url": self.weaviate_url,
                    "api_key": self.weaviate_api_key,
                    "class_name": self.weaviate_class,
                    "vectorizer": self.weaviate_vectorizer,
                }
            )

        return config
