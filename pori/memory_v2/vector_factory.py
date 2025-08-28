"""
Vector store factory for creating different vector database implementations.

This factory pattern allows easy swapping between different vector databases
without coupling the memory system to specific implementations.
"""

from typing import Dict, Any, Optional
from .stores.vector_store import VectorMemoryStore
from .memory_store import MemoryStore


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create_vector_store(
        backend: str = "local", config: Optional[Dict[str, Any]] = None
    ) -> MemoryStore[str]:
        """Create a vector store instance based on backend type.

        Args:
            backend: Vector store backend type ("local", "weaviate", etc.)
            config: Configuration dictionary for the specific backend

        Returns:
            MemoryStore instance for the specified backend

        Raises:
            ValueError: If backend type is not supported
            ImportError: If required dependencies are not installed
        """
        config = config or {}

        if backend.lower() == "local":
            return VectorMemoryStore(
                model_name=config.get("model_name", "all-MiniLM-L6-v2")
            )

        elif backend.lower() == "weaviate":
            try:
                from .stores.weaviate_store import WeaviateMemoryStore

                return WeaviateMemoryStore(
                    url=config.get("url"),
                    api_key=config.get("api_key"),
                    class_name=config.get("class_name", "AgentMemory"),
                    **config.get("extra_kwargs", {}),
                )
            except ImportError as e:
                raise ImportError(
                    f"Weaviate dependencies not installed. Install with: pip install weaviate-client\n"
                    f"Original error: {e}"
                )

        # Future vector stores can be added here:
        # elif backend.lower() == "pinecone":
        #     from .stores.pinecone_store import PineconeMemoryStore
        #     return PineconeMemoryStore(...)
        # elif backend.lower() == "chroma":
        #     from .stores.chroma_store import ChromaMemoryStore
        #     return ChromaMemoryStore(...)

        else:
            raise ValueError(
                f"Unsupported vector backend: {backend}. "
                f"Supported backends: local, weaviate"
            )


def create_vector_store(backend: str, **kwargs) -> MemoryStore[str]:
    """Convenience function for creating vector stores.

    Args:
        backend: Vector store backend type
        **kwargs: Configuration parameters for the backend

    Returns:
        MemoryStore instance
    """
    return VectorStoreFactory.create_vector_store(backend, kwargs)

