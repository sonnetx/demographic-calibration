"""Verbalized confidence extraction."""

from __future__ import annotations

import re


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> blocks from reasoning model output.

    If all content is inside thinking tags (nothing meaningful after </think>),
    falls back to using the content inside the tags so answers aren't lost.
    """
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    # All content was inside <think> tags — extract it rather than returning empty
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # No think tags at all
    return text.strip()


_CONFIDENCE_PATTERNS = [
    # "Confidence: 85%" / "Confidence: 85"
    r"[Cc]onfidence[:\s]+(\d+(?:\.\d+)?)\s*%",
    # "I am 85% confident"
    r"(\d+(?:\.\d+)?)\s*%\s*confident",
    # "85% confidence" (number before word, e.g. OpenFlamingo)
    r"(\d+(?:\.\d+)?)\s*%\s*confidence",
    # "confidence of 85%"
    r"confidence\s+of\s+(\d+(?:\.\d+)?)\s*%",
    # "85% certainty"
    r"(\d+(?:\.\d+)?)\s*%\s*certain",
    # "85 percent confident"
    r"(\d+(?:\.\d+)?)\s*percent\s*confident",
    # "85% probability/chance/likely/sure"
    r"(\d+(?:\.\d+)?)\s*%\s*(?:probability|probable|chance|likely|likelihood|sure)",
    # "probability: 85%" / "probability of 85%"
    r"probability[^0-9\n]*(\d+(?:\.\d+)?)\s*%?",
    # "85 percent" (without requiring "confident" after)
    r"(\d+(?:\.\d+)?)\s*percent",
    # "85/100" or "85 out of 100"
    r"(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*100",
    # Standalone percentage near "confidence" on same line
    r"[Cc]onfidence[^0-9\n]*(\d+(?:\.\d+)?)\s*%?",
    # Bare decimal between 0 and 1 near "confidence" (e.g. "confidence is 0.85")
    r"[Cc]onfidence[^0-9\n]*(0\.\d+)",
    # Any percentage in the response (last resort, broad match)
    r"(\d+(?:\.\d+)?)\s*%",
    # Standalone percentage at start of response (prompt ends with "Confidence:")
    r"^\s*(\d+(?:\.\d+)?)\s*%",
]


def _match_confidence_patterns(text: str) -> float | None:
    """Try all confidence patterns against text, return first match or None."""
    for pattern in _CONFIDENCE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                # Normalize to [0, 1]
                if value > 1:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
            except ValueError:
                continue
    return None


def extract_verbalized_confidence(response_text: str) -> float | None:
    """
    Extract verbalized confidence from model response.

    Parses confidence percentage from model's self-reported confidence.
    Uses a two-pass strategy for reasoning models:
      1. Try the text after </think> (the actual answer).
      2. Fall back to searching inside <think> blocks, where reasoning
         models often place numeric confidence before the final answer.

    Args:
        response_text: Full model response text

    Returns:
        Confidence score in [0, 1], or None if not found
    """
    # Pass 1: try text outside <think> blocks (the actual answer)
    stripped = _strip_thinking_tags(response_text)
    result = _match_confidence_patterns(stripped)
    if result is not None:
        return result

    # Pass 2: search inside <think> content (R1 often puts confidence
    # reasoning here but only outputs the classification after </think>)
    think_match = re.search(r"<think>(.*?)</think>", response_text, flags=re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        result = _match_confidence_patterns(think_content)
        if result is not None:
            return result

    return None


def extract_confidence_with_context(response_text: str) -> tuple[float | None, str]:
    """
    Extract confidence and the surrounding context.

    Args:
        response_text: Full model response text

    Returns:
        Tuple of (confidence, context_string)
    """
    confidence = extract_verbalized_confidence(response_text)

    # Extract surrounding context
    context = ""
    if confidence is not None:
        # Find the sentence containing confidence
        sentences = re.split(r"[.!?\n]", response_text)
        for sentence in sentences:
            if re.search(r"\d+\s*%", sentence):
                context = sentence.strip()
                break

    return confidence, context


def extract_all_percentages(response_text: str) -> list[float]:
    """
    Extract all percentage values from response.

    Useful for debugging or when multiple confidence values are present.

    Args:
        response_text: Full model response text

    Returns:
        List of percentage values (normalized to [0, 1])
    """
    pattern = r"(\d+(?:\.\d+)?)\s*%"
    matches = re.findall(pattern, response_text)

    percentages = []
    for match in matches:
        try:
            value = float(match) / 100.0
            percentages.append(max(0.0, min(1.0, value)))
        except ValueError:
            continue

    return percentages
