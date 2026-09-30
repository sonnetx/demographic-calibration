"""Embedding-based sample consistency confidence extraction.

Implements the Sample Consistency approach: generate K stochastic responses,
embed each with an embedding model, and compute average pairwise cosine
similarity. High similarity = high confidence.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from ..data.dataset import Sample
from ..models.base import BaseVLM

logger = logging.getLogger(__name__)


async def get_openai_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
) -> np.ndarray:
    """
    Get embeddings from OpenAI embedding API.

    Args:
        texts: List of texts to embed
        model: Embedding model name
        api_key: OpenAI API key (uses OPENAI_API_KEY env if None)

    Returns:
        Array of shape (len(texts), embedding_dim)
    """
    import os

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = await client.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array of shape (n, d)

    Returns:
        Similarity matrix of shape (n, n)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    normalized = embeddings / norms
    return normalized @ normalized.T


def average_pairwise_cosine_similarity(embeddings: np.ndarray) -> float:
    """
    Compute average pairwise cosine similarity across all pairs.

    Args:
        embeddings: Array of shape (n, d)

    Returns:
        Average cosine similarity (scalar)
    """
    sim_matrix = cosine_similarity_matrix(embeddings)
    n = len(sim_matrix)
    if n < 2:
        return 1.0

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[triu_indices]
    return float(np.mean(pairwise_sims))


async def compute_embedding_consistency(
    model: BaseVLM,
    sample: Sample,
    prompt: str,
    k: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 256,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_key: str | None = None,
) -> tuple[float, list[str]]:
    """
    Compute embedding-based sample consistency confidence.

    Generates K stochastic responses, embeds each using an embedding model,
    and returns the average pairwise cosine similarity as the confidence score.

    Args:
        model: VLM model instance
        sample: Data sample
        prompt: Diagnosis prompt
        k: Number of generations
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        embedding_model: Name of the embedding model to use
        embedding_api_key: API key for the embedding model

    Returns:
        Tuple of (average_cosine_similarity, list_of_response_texts)
    """
    image_base64 = sample.to_base64()

    # Generate K responses at temperature > 0
    tasks = [
        model.predict(
            image_base64=image_base64,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            return_logprobs=False,
        )
        for _ in range(k)
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors, collect response texts
    from ..models.base import VLMResponse

    valid_texts = [r.text for r in responses if isinstance(r, VLMResponse)]
    n_errors = len(responses) - len(valid_texts)

    if n_errors > 0:
        logger.warning(
            "Embedding consistency: %d/%d samples failed for %s",
            n_errors, k, sample.image_id,
        )

    if len(valid_texts) < 2:
        raise RuntimeError(
            f"Need at least 2 valid responses for embedding consistency, got {len(valid_texts)}"
        )

    # Embed all responses
    embeddings = await get_openai_embeddings(
        valid_texts, model=embedding_model, api_key=embedding_api_key
    )

    # Compute average pairwise cosine similarity
    avg_sim = average_pairwise_cosine_similarity(embeddings)

    logger.info(
        "Embedding consistency [%s]: %d/%d valid, avg_cosine_sim=%.4f",
        sample.image_id, len(valid_texts), k, avg_sim,
    )

    return avg_sim, valid_texts
