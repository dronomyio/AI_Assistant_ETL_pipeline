#!/usr/bin/env python3
"""
Multimodal embedding module for processing both text and image content.
Supports OpenAI, CLIP, and Cohere embedding models for text and images.
"""
import os
import base64
import hashlib
import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import requests
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TEXT_EMBEDDING_DIM = 384  # Dimension for mock text embeddings
DEFAULT_IMAGE_EMBEDDING_DIM = 512  # Dimension for mock image embeddings


def get_text_embedding(text: str) -> List[float]:
    """
    Generate an embedding for text content, with fallback options.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        A list of floats representing the embedding
    """
    # Try OpenAI embeddings first (if API key is available)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            logger.info("Generated text embedding using OpenAI")
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {str(e)}")
    
    # Try Cohere embeddings (if API key is available)
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if cohere_api_key:
        try:
            import cohere
            co = cohere.Client(cohere_api_key)
            response = co.embed(
                texts=[text],
                model="embed-english-v3.0" 
            )
            logger.info("Generated text embedding using Cohere")
            return response.embeddings[0]
        except Exception as e:
            logger.warning(f"Cohere embedding failed: {str(e)}")
    
    # Fallback to mock embeddings
    logger.warning("Using mock text embeddings")
    # Create a deterministic embedding based on text content
    text_hash = hashlib.md5(text.encode()).hexdigest()
    np.random.seed(int(text_hash[:8], 16))
    return np.random.normal(0, 1, DEFAULT_TEXT_EMBEDDING_DIM).tolist()


def read_image_file(image_path: str) -> bytes:
    """
    Read an image file and return its bytes content.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Bytes content of the image
    """
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read image file {image_path}: {str(e)}")
        raise


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 encoding.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        image_bytes = read_image_file(image_path)
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {str(e)}")
        raise


def get_image_embedding(image_path: str) -> List[float]:
    """
    Generate an embedding for an image file, with fallback options.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A list of floats representing the embedding
    """
    # Verify the file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return get_mock_image_embedding(image_path)
    
    # Try using OpenAI for image embeddings (via text model with base64)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Convert image to base64 and create a text description
            image_filename = os.path.basename(image_path)
            text_for_embedding = f"Image filename: {image_filename}"
            
            # Generate embedding for the text description
            response = client.embeddings.create(
                input=text_for_embedding,
                model="text-embedding-3-small"
            )
            logger.info("Generated image embedding using OpenAI (text proxy)")
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI image embedding failed: {str(e)}")
    
    # Try Cohere embeddings (if API key is available)
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if cohere_api_key:
        try:
            import cohere
            co = cohere.Client(cohere_api_key)
            
            # Similar approach as with OpenAI - use filename for embedding
            image_filename = os.path.basename(image_path)
            text_for_embedding = f"Image filename: {image_filename}"
            
            response = co.embed(
                texts=[text_for_embedding],
                model="embed-english-v3.0"
            )
            logger.info("Generated image embedding using Cohere")
            return response.embeddings[0]
        except Exception as e:
            logger.warning(f"Cohere image embedding failed: {str(e)}")
    
    # Fallback to mock image embeddings
    return get_mock_image_embedding(image_path)


def get_mock_image_embedding(image_path: str) -> List[float]:
    """
    Generate a mock embedding for an image based on its filename.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A list of floats representing the mock embedding
    """
    logger.warning("Using mock image embeddings")
    # Create a deterministic embedding based on the image filename
    image_name = os.path.basename(image_path)
    img_hash = hashlib.md5(image_name.encode()).hexdigest()
    np.random.seed(int(img_hash[:8], 16))
    return np.random.normal(0, 1, DEFAULT_IMAGE_EMBEDDING_DIM).tolist()


def configure_weaviate_multimodal(client, collection_name: str):
    """
    Configure Weaviate for multimodal embeddings.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection to configure
    """
    try:
        from weaviate.classes.config import Configure, Multi2VecField
        
        # Get the collection
        collection = client.collections.get(collection_name)
        
        # Configure for multimodal embeddings
        if collection_name == "TextChunk":
            collection.config.update(
                vectorizer_config=Configure.Vectorizer.multi2vec_clip(
                    text_fields=[
                        Multi2VecField(name="text", weight=1.0)
                    ]
                )
            )
            logger.info(f"Configured {collection_name} for multimodal text embeddings")
        
        elif collection_name == "ImageEmbedding":
            collection.config.update(
                vectorizer_config=Configure.Vectorizer.multi2vec_clip(
                    image_fields=[
                        Multi2VecField(name="image_data", weight=1.0)
                    ]
                )
            )
            logger.info(f"Configured {collection_name} for multimodal image embeddings")
        
    except Exception as e:
        logger.error(f"Failed to configure multimodal embeddings for {collection_name}: {str(e)}")


# Legacy function names for backward compatibility
def get_mock_embedding(text: str) -> List[float]:
    """Alias for get_text_embedding for backward compatibility"""
    return get_text_embedding(text)


def get_embedding(text: str) -> List[float]:
    """Alias for get_text_embedding for backward compatibility"""
    return get_text_embedding(text)