# Vector Store Architecture

## Overview

The Pori agent system now supports multiple vector databases through a clean factory pattern that allows easy swapping between different implementations without tight coupling.

## Architecture

### Factory Pattern

The `VectorStoreFactory` provides a clean interface for creating different vector store implementations:

```python
from pori.memory_v2.vector_factory import VectorStoreFactory

# Create Weaviate vector store
weaviate_store = VectorStoreFactory.create_vector_store("weaviate", {
    "url": "http://localhost:8080",
    "class_name": "MyMemories"
})
```

### Supported Backends

1. **Weaviate** (`"weaviate"`): Uses Weaviate's built-in vectorization

## Configuration

### Environment Variables

```bash
# Vector backend selection
VECTOR=true
VECTOR_BACKEND=weaviate

# Weaviate configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-api-key  # Optional, for Weaviate Cloud
WEAVIATE_CLASS=AgentMemory      # Optional, defaults to AgentMemory
WEAVIATE_VECTORIZER=text2vec-openai  # Optional, defaults to text2vec-openai

# If using OpenAI vectorizer
OPENAI_API_KEY=your-openai-key
```

### Programmatic Configuration

```python
# Weaviate vector store
memory_config = {
    "vector": True,
    "vector_backend": "weaviate",
    "weaviate_url": "http://localhost:8080",
    "weaviate_class": "AgentMemory",
    "weaviate_vectorizer": "text2vec-openai"
}

orchestrator = Orchestrator(llm=llm, tools_registry=registry, memory_config=memory_config)
```

## Weaviate Setup

### Local Weaviate with Docker

```bash
# Basic Weaviate with OpenAI vectorizer
docker run -d \
  -p 8080:8080 \
  -e OPENAI_APIKEY=$OPENAI_API_KEY \
  -e ENABLE_MODULES=text2vec-openai \
  cr.weaviate.io/semitechnologies/weaviate:1.22.4
```

### Weaviate Cloud

1. Sign up at [Weaviate Cloud](https://console.weaviate.cloud/)
2. Create a cluster
3. Get your cluster URL and API key
4. Set environment variables:
   ```bash
   WEAVIATE_URL=https://your-cluster.weaviate.network
   WEAVIATE_API_KEY=your-weaviate-api-key
   ```

## Benefits

### Decoupling
- **No tight coupling**: Vector stores are created through factory pattern
- **Easy swapping**: Change `VECTOR_BACKEND` environment variable to switch
- **Clean interfaces**: All vector stores implement the same `MemoryStore` interface

### Weaviate Advantages
- **Built-in vectorization**: Uses Weaviate's optimized vectorizers (no SentenceTransformer overhead)
- **Persistence**: Memories survive application restarts
- **Scalability**: Can handle large amounts of data
- **Multiple vectorizers**: Support for OpenAI, Cohere, Hugging Face, etc.

```

## Adding New Vector Stores

To add a new vector database (e.g., Pinecone, Chroma):

1. **Create the store implementation**:
   ```python
   # pori/memory_v2/stores/pinecone_store.py
   class PineconeMemoryStore(MemoryStore[str]):
       def __init__(self, api_key: str, environment: str):
           # Implementation
   ```

2. **Add to factory**:
   ```python
   # pori/memory_v2/vector_factory.py
   elif backend.lower() == "pinecone":
       from .stores.pinecone_store import PineconeMemoryStore
       return PineconeMemoryStore(
           api_key=config.get("api_key"),
           environment=config.get("environment")
       )
   ```

3. **Update configuration handling** in `orchestrator.py`

## Migration Guide

### From Old Tightly-Coupled Approach

**Before** (tightly coupled):
```python
memory = EnhancedAgentMemory(
    vector_backend="weaviate",
    weaviate_url="http://localhost:8080"
)
```

**After** (factory pattern):
```python
memory = EnhancedAgentMemory(
    vector_config={
        "backend": "weaviate",
        "url": "http://localhost:8080"
    }
)
```

### Environment Variables

The environment variables remain the same, but now they're processed through the factory pattern for better separation of concerns.

## Best Practices

1. **Use Weaviate for production**: Persistent and scalable
2. **Configure via environment**: Makes deployment easier
3. **Test with different backends**: Ensure your code works with any vector store



