"""File I/O utilities."""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..confidence.aggregator import ConfidenceResult


def save_results(results: list["ConfidenceResult"], path: Path) -> None:
    """
    Save confidence results to JSON file.

    Args:
        results: List of ConfidenceResult objects
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: Path) -> list[dict]:
    """
    Load results from JSON file.

    Args:
        path: Input file path

    Returns:
        List of result dictionaries
    """
    with open(path) as f:
        return json.load(f)


logger = logging.getLogger(__name__)


def append_checkpoint(result: "ConfidenceResult", path: Path) -> None:
    """Append a single ConfidenceResult as a JSON line to a checkpoint file.

    Each line is a self-contained JSON object. Uses flush + fsync so a
    crash loses at most the in-flight sample, never a completed one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint(path: Path) -> list[dict]:
    """Load completed results from a JSONL checkpoint file.

    Tolerates a truncated final line (from a crash mid-write) by
    discarding it with a warning.

    Returns:
        List of result dictionaries, one per valid line.
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Discarding corrupted checkpoint line %d in %s",
                    line_num,
                    path,
                )
    return results


def checkpoint_to_results(checkpoint_path: Path, results_path: Path) -> None:
    """Convert a JSONL checkpoint into the final results.json format.

    Writes results.json first, then deletes the checkpoint. If the
    process dies between the two operations, the checkpoint survives
    and will be picked up on the next resume.
    """
    records = load_checkpoint(checkpoint_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    checkpoint_path.unlink()
