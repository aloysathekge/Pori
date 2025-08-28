from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
import logging

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery

from ..memory_store import MemoryStore

logger = logging.getLogger("pori.memory.weaviate")


class WeaviateMemoryStore(MemoryStore[str]):
    """Weaviate vector store implementation for agent memory.

    Features:
    - Persistent storage in Weaviate
    - Semantic search using vector embeddings
    - Metadata support
    - Configurable embedding model
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        class_name: str = "AgentMemory",
        vectorizer: str = "text2vec-huggingface",  # Use free HuggingFace vectorizer by default
        **weaviate_kwargs,
    ) -> None:
        """Initialize Weaviate connection using built-in vectorization.

        Args:
            url: Weaviate instance URL (defaults to env WEAVIATE_URL or localhost)
            api_key: Weaviate API key (defaults to env WEAVIATE_API_KEY)
            class_name: Weaviate class name for storing memories
            vectorizer: Weaviate vectorizer module to use (e.g., text2vec-openai, text2vec-cohere)
            **weaviate_kwargs: Additional arguments for Weaviate client
        """
        self.class_name = class_name
        self.vectorizer = vectorizer

        # Get connection details from environment if not provided
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")

        # Initialize Weaviate client
        self._init_client(**weaviate_kwargs)
        self._ensure_schema()

    def _init_client(self, **kwargs) -> None:
        """Initialize the Weaviate client with proper authentication."""
        try:
            if self.api_key:
                # Cloud/authenticated instance with extended timeout
                import weaviate.classes as wvc

                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                    additional_config=wvc.init.AdditionalConfig(
                        timeout=wvc.init.Timeout(
                            init=30, query=60, insert=60
                        )  # Extended timeouts
                    ),
                    **kwargs,
                )
            else:
                # Local instance
                self.client = weaviate.connect_to_local(
                    host=self.url.replace("http://", "").replace("https://", ""),
                    **kwargs,
                )
            logger.info(f"Connected to Weaviate at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def _ensure_schema(self) -> None:
        """Ensure the required class exists in Weaviate schema."""
        try:
            collections = self.client.collections

            # Check if collection exists and has the right vectorizer
            if collections.exists(self.class_name):
                # For now, let's delete and recreate to ensure correct vectorizer
                logger.info(f"Deleting existing collection: {self.class_name}")
                collections.delete(self.class_name)

            # Create collection with Weaviate's built-in vectorizer
            vectorizer_config = self._get_vectorizer_config()
            collections.create(
                name=self.class_name,
                vectorizer_config=vectorizer_config,  # Keep using vectorizer_config for now
                properties=[
                    weaviate.classes.config.Property(
                        name="content",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="The memory content/text",
                    ),
                    weaviate.classes.config.Property(
                        name="key",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Unique key for the memory item",
                    ),
                    weaviate.classes.config.Property(
                        name="importance",
                        data_type=weaviate.classes.config.DataType.NUMBER,
                        description="Importance score of the memory",
                    ),
                    weaviate.classes.config.Property(
                        name="memory_type",
                        data_type=weaviate.classes.config.DataType.TEXT,
                        description="Type of memory (task, summary, etc.)",
                    ),
                ],
            )
            logger.info(
                f"Created Weaviate collection: {self.class_name} with vectorizer: {self.vectorizer}"
            )

        except Exception as e:
            logger.error(f"Failed to ensure Weaviate schema: {e}")
            raise

    def _get_vectorizer_config(self):
        """Get vectorizer configuration based on the selected vectorizer."""
        if self.vectorizer == "text2vec-openai":
            return Configure.Vectorizer.text2vec_openai()
        elif self.vectorizer == "text2vec-cohere":
            return Configure.Vectorizer.text2vec_cohere()
        elif self.vectorizer == "text2vec-huggingface":
            return Configure.Vectorizer.text2vec_huggingface()
        elif self.vectorizer == "none":
            # For cases where you want to provide your own vectors
            return Configure.Vectorizer.none()
        else:
            logger.warning(
                f"Unknown vectorizer {self.vectorizer}, falling back to text2vec-openai"
            )
            return Configure.Vectorizer.text2vec_openai()

    def add(self, key: str, value: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add or update a memory item in Weaviate."""
        try:
            collection = self.client.collections.get(self.class_name)

            # Prepare data object with flattened metadata
            meta = meta or {}
            data_object = {
                "content": value,
                "key": key,
                "importance": meta.get("importance", 1),
                "memory_type": meta.get("type", "general"),
            }

            # Check if key already exists and delete it
            existing = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("key").equal(key),
                limit=1,
            )

            if existing.objects:
                # Delete existing object
                collection.data.delete_by_id(existing.objects[0].uuid)
                logger.debug(f"Deleted existing memory with key: {key}")

            # Insert new object - Weaviate will automatically vectorize the content
            uuid_obj = collection.data.insert(properties=data_object)

            logger.debug(f"Added memory to Weaviate: key={key}, uuid={uuid_obj}")

        except Exception as e:
            logger.error(f"Failed to add memory to Weaviate: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """Retrieve a memory item by key."""
        try:
            collection = self.client.collections.get(self.class_name)

            result = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("key").equal(key),
                limit=1,
            )

            if result.objects:
                return result.objects[0].properties.get("content")
            return None

        except Exception as e:
            logger.error(f"Failed to get memory from Weaviate: {e}")
            return None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for similar memories using vector similarity."""
        if not query:
            return []

        try:
            collection = self.client.collections.get(self.class_name)

            # Use Weaviate's built-in near_text search for automatic vectorization
            result = collection.query.near_text(
                query=query,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
            )

            # Format results
            results: List[Tuple[str, str, float]] = []
            for obj in result.objects:
                key = obj.properties.get("key", "")
                content = obj.properties.get("content", "")
                # Convert distance to similarity score (1 - distance)
                distance = obj.metadata.distance or 0.0
                similarity = max(0.0, 1.0 - distance)
                results.append((key, content, similarity))

            logger.debug(
                f"Vector search returned {len(results)} results for query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search Weaviate: {e}")
            return []

    def forget(self, key: str) -> bool:
        """Delete a memory item by key."""
        try:
            collection = self.client.collections.get(self.class_name)

            # Find the object by key
            result = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("key").equal(key),
                limit=1,
            )

            if result.objects:
                # Delete the object
                collection.data.delete_by_id(result.objects[0].uuid)
                logger.debug(f"Deleted memory with key: {key}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete memory from Weaviate: {e}")
            return False

    def close(self) -> None:
        """Close the Weaviate connection."""
        try:
            if hasattr(self, "client"):
                self.client.close()
                logger.info("Closed Weaviate connection")
        except Exception as e:
            logger.error(f"Error closing Weaviate connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
