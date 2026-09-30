"""DDI Dataset loader."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import pandas as pd
from PIL import Image


@runtime_checkable
class Sample(Protocol):
    """Protocol for any dataset sample used in the calibration pipeline."""

    image_id: str
    image_path: Path
    label: int  # 0 or 1 (binary)
    label_name: str  # human-readable label (e.g. "benign", "present")
    demographic_group: str  # e.g. "light", "male"

    def load_image(self) -> Image.Image: ...
    def to_base64(self) -> str: ...


@dataclass
class DDISample:
    """Single DDI dataset sample."""

    image_id: str
    image_path: Path
    diagnosis: int  # 0=benign, 1=malignant
    diagnosis_label: str  # "benign" or "malignant"
    skin_tone: int  # FST grouping (e.g., 12, 34, 56)
    skin_tone_group: str  # "light", "dark", etc.

    @property
    def label(self) -> int:
        return self.diagnosis

    @property
    def label_name(self) -> str:
        return self.diagnosis_label

    @property
    def demographic_group(self) -> str:
        return self.skin_tone_group

    def load_image(self) -> Image.Image:
        """Load and return PIL Image."""
        return Image.open(self.image_path).convert("RGB")

    def to_base64(self) -> str:
        """
        Convert image to base64 string for API calls.

        Returns:
            Base64-encoded JPEG string
        """
        img = self.load_image()
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class DDIDataset:
    """DDI Dataset loader."""

    DIAGNOSIS_MAP = {0: "benign", 1: "malignant"}
    DEFAULT_SKIN_TONE_GROUPS = {"light": [12], "medium": [34], "dark": [56]}

    def __init__(
        self,
        metadata_path: Path,
        images_dir: Path,
        skin_tone_groups: dict[str, list[int]] | None = None,
    ):
        """
        Initialize DDI dataset.

        Args:
            metadata_path: Path to ddi_metadata.csv
            images_dir: Path to images directory
            skin_tone_groups: Custom skin tone groupings (default: light=[12], dark=[56])
        """
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.skin_tone_groups = skin_tone_groups or self.DEFAULT_SKIN_TONE_GROUPS

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load and validate metadata CSV."""
        self.metadata = pd.read_csv(self.metadata_path)

        # Check for required columns - try multiple possible column names
        # DDI dataset may use different column names
        file_col = self._find_column(["DDI_file", "DDI_ID", "image_id", "filename"])
        malignancy_col = self._find_column(["malignant", "label", "diagnosis"])
        skin_tone_col = self._find_column(["skin_tone", "fitzpatrick", "fst", "FST"])

        if not all([file_col, malignancy_col, skin_tone_col]):
            available = list(self.metadata.columns)
            raise ValueError(
                f"Missing required columns. Available columns: {available}. "
                f"Need columns for: file ID, malignancy label, skin tone."
            )

        # Normalize column names
        self._file_col = file_col
        self._malignancy_col = malignancy_col
        self._skin_tone_col = skin_tone_col

        # Create diagnosis column (ensure binary 0/1)
        malignancy_values = self.metadata[malignancy_col]
        if malignancy_values.dtype == bool:
            self.metadata["_diagnosis"] = malignancy_values.astype(int)
        elif malignancy_values.dtype == object:
            # Handle string labels
            self.metadata["_diagnosis"] = malignancy_values.str.lower().map(
                {"malignant": 1, "benign": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)
        else:
            self.metadata["_diagnosis"] = (malignancy_values == 1).astype(int)

        # Create reverse mapping for skin tone groups
        self._skin_tone_to_group: dict[int, str] = {}
        for group_name, codes in self.skin_tone_groups.items():
            for code in codes:
                self._skin_tone_to_group[code] = group_name

        # Filter out samples with skin tones not in any defined group
        valid_tones = set(self._skin_tone_to_group.keys())
        before = len(self.metadata)
        self.metadata = self.metadata[
            self.metadata[self._skin_tone_col].isin(valid_tones)
        ].reset_index(drop=True)
        after = len(self.metadata)
        if before != after:
            import logging
            logging.getLogger(__name__).info(
                "Filtered out %d samples with unmapped skin tones (kept %d/%d)",
                before - after, after, before,
            )

    def _find_column(self, candidates: list[str]) -> str | None:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in self.metadata.columns:
                return col
            # Try case-insensitive match
            for actual_col in self.metadata.columns:
                if actual_col.lower() == col.lower():
                    return actual_col
        return None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> DDISample:
        row = self.metadata.iloc[idx]
        skin_tone = int(row[self._skin_tone_col])
        diagnosis = int(row["_diagnosis"])

        return DDISample(
            image_id=str(row[self._file_col]),
            image_path=self.images_dir / str(row[self._file_col]),
            diagnosis=diagnosis,
            diagnosis_label=self.DIAGNOSIS_MAP[diagnosis],
            skin_tone=skin_tone,
            skin_tone_group=self._skin_tone_to_group.get(skin_tone, "unknown"),
        )

    def __iter__(self) -> Iterator[DDISample]:
        for idx in range(len(self)):
            yield self[idx]

    def filter_by_skin_tone_group(self, group: str) -> list[DDISample]:
        """
        Filter samples by skin tone group.

        Args:
            group: Group name (e.g., 'light' or 'dark')

        Returns:
            List of samples in that group
        """
        if group not in self.skin_tone_groups:
            raise ValueError(
                f"Unknown group '{group}'. Available: {list(self.skin_tone_groups.keys())}"
            )

        valid_codes = set(self.skin_tone_groups[group])
        return [s for s in self if s.skin_tone in valid_codes]

    def get_demographic_subsets(self) -> dict[str, list[DDISample]]:
        """
        Get samples grouped by demographic.

        Returns:
            Dictionary mapping group names to sample lists
        """
        return {group: self.filter_by_skin_tone_group(group) for group in self.skin_tone_groups}

    def get_group_statistics(self) -> pd.DataFrame:
        """
        Get statistics for each demographic group.

        Returns:
            DataFrame with group-level statistics
        """
        stats = []
        for group, samples in self.get_demographic_subsets().items():
            n_total = len(samples)
            n_malignant = sum(1 for s in samples if s.diagnosis == 1)
            n_benign = n_total - n_malignant

            stats.append({
                "group": group,
                "n_total": n_total,
                "n_benign": n_benign,
                "n_malignant": n_malignant,
                "malignant_rate": n_malignant / n_total if n_total > 0 else 0,
            })

        return pd.DataFrame(stats)
