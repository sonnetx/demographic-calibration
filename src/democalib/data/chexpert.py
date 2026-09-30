"""CheXpert Dataset loader for binary condition classification."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "absent", 1: "present"}


@dataclass
class CheXpertSample:
    """Single CheXpert dataset sample for one condition."""

    image_id: str
    image_path: Path
    label: int  # 0=absent, 1=present
    label_name: str  # "absent" or "present"
    demographic_group: str  # "male" or "female"
    sex: str  # raw sex value from CSV

    def load_image(self) -> Image.Image:
        """Load and return PIL Image."""
        return Image.open(self.image_path).convert("RGB")

    def to_base64(self) -> str:
        """Convert image to base64 string for API calls."""
        img = self.load_image()
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class CheXpertDataset:
    """CheXpert dataset loader for binary condition classification.

    Loads the CheXpert CSV metadata, extracts a single condition column
    as binary label, and groups by sex for demographic analysis.
    """

    ALL_CONDITIONS = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    DEFAULT_DEMOGRAPHIC_GROUPS = {"male": ["Male"], "female": ["Female"]}

    def __init__(
        self,
        metadata_path: Path,
        images_dir: Path,
        condition: str = "Pneumonia",
        demographic_groups: dict[str, list[str]] | None = None,
    ):
        """
        Initialize CheXpert dataset.

        Args:
            metadata_path: Path to train_valid_combined.csv
            images_dir: Path to flat image directory
            condition: Which condition column to use as binary label
            demographic_groups: Mapping of group names to Sex values
        """
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.condition = condition
        self.demographic_groups = demographic_groups or self.DEFAULT_DEMOGRAPHIC_GROUPS

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load and validate metadata CSV."""
        self.metadata = pd.read_csv(self.metadata_path)

        # Validate condition column exists
        if self.condition not in self.metadata.columns:
            available = [c for c in self.metadata.columns if c in self.ALL_CONDITIONS]
            raise ValueError(
                f"Condition '{self.condition}' not found. "
                f"Available conditions: {available}"
            )

        # Fill NaN and -1 (uncertain) with 0 (absent)
        self.metadata["_label"] = (
            self.metadata[self.condition]
            .fillna(0)
            .replace(-1, 0)
            .astype(int)
            .clip(0, 1)
        )

        # Find sex column
        sex_col = self._find_column(["Sex", "sex", "gender", "Gender"])
        if sex_col is None:
            raise ValueError(
                f"Sex column not found. Available columns: {list(self.metadata.columns)}"
            )
        self._sex_col = sex_col

        # Find image_id column (prefer image_id, fall back to Path)
        id_col = self._find_column(["image_id", "Image_ID"])
        path_col = self._find_column(["Path", "path"])
        if id_col is not None:
            self._id_col = id_col
        elif path_col is not None:
            # Derive flat filename from hierarchical path
            self.metadata["_image_id"] = self.metadata[path_col].apply(
                self._hierarchical_to_flat
            )
            self._id_col = "_image_id"
        else:
            raise ValueError("No image ID or Path column found.")

        # Build demographic group mapping
        self._sex_to_group: dict[str, str] = {}
        for group_name, values in self.demographic_groups.items():
            for v in values:
                self._sex_to_group[v] = group_name

        # Filter to samples with valid sex values
        valid_sex = set(self._sex_to_group.keys())
        before = len(self.metadata)
        self.metadata = self.metadata[
            self.metadata[self._sex_col].isin(valid_sex)
        ].reset_index(drop=True)
        after = len(self.metadata)
        if before != after:
            logger.info(
                "Filtered out %d samples with unmapped sex values (kept %d/%d)",
                before - after, after, before,
            )

    def _find_column(self, candidates: list[str]) -> str | None:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in self.metadata.columns:
                return col
            for actual_col in self.metadata.columns:
                if actual_col.lower() == col.lower():
                    return actual_col
        return None

    @staticmethod
    def _hierarchical_to_flat(path_str: str) -> str:
        """Convert hierarchical CheXpert path to flat filename.

        e.g. "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
          -> "patient00001_study1_view1_frontal.jpg"
        """
        parts = Path(path_str).parts
        # Keep patient/study/view parts, skip dataset version and split prefixes
        relevant = [
            p for p in parts
            if p.startswith("patient") or p.startswith("study") or p.endswith(".jpg")
        ]
        if len(relevant) >= 3:
            return f"{relevant[0]}_{relevant[1]}_{relevant[2]}"
        # Fallback: replace path separators
        return path_str.replace("/", "_").replace("\\", "_")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> CheXpertSample:
        row = self.metadata.iloc[idx]
        label = int(row["_label"])
        sex = str(row[self._sex_col])
        image_id = str(row[self._id_col])

        return CheXpertSample(
            image_id=image_id,
            image_path=self.images_dir / image_id,
            label=label,
            label_name=LABEL_MAP[label],
            demographic_group=self._sex_to_group.get(sex, "unknown"),
            sex=sex,
        )

    def __iter__(self) -> Iterator[CheXpertSample]:
        for idx in range(len(self)):
            yield self[idx]

    def filter_by_demographic_group(self, group: str) -> list[CheXpertSample]:
        """Filter samples by demographic group."""
        if group not in self.demographic_groups:
            raise ValueError(
                f"Unknown group '{group}'. Available: {list(self.demographic_groups.keys())}"
            )
        return [s for s in self if s.demographic_group == group]

    def get_demographic_subsets(self) -> dict[str, list[CheXpertSample]]:
        """Get samples grouped by demographic."""
        return {
            group: self.filter_by_demographic_group(group)
            for group in self.demographic_groups
        }

    def get_group_statistics(self) -> pd.DataFrame:
        """Get statistics for each demographic group."""
        stats = []
        for group, samples in self.get_demographic_subsets().items():
            n_total = len(samples)
            n_present = sum(1 for s in samples if s.label == 1)
            n_absent = n_total - n_present

            stats.append({
                "group": group,
                "n_total": n_total,
                "n_absent": n_absent,
                "n_present": n_present,
                "present_rate": n_present / n_total if n_total > 0 else 0,
            })

        return pd.DataFrame(stats)
