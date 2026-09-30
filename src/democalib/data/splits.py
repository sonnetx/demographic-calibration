"""Dataset splitting utilities."""

from sklearn.model_selection import train_test_split

from .dataset import DDIDataset, DDISample


def create_stratified_splits(
    dataset: DDIDataset,
    val_ratio: float = 0.20,
    random_seed: int = 42,
) -> tuple[list[DDISample], list[DDISample]]:
    """
    Create stratified val/test splits.

    Stratification is performed on (diagnosis, skin_tone) pairs to maintain
    demographic balance across splits. The val set is used for fitting
    temperature scaling; the test set is used for evaluation.

    Args:
        dataset: DDI dataset instance
        val_ratio: Fraction of data used for validation (rest is test)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (val_samples, test_samples)
    """
    all_samples = list(dataset)

    # Create stratification key combining diagnosis and skin tone
    stratify_keys = [f"{s.diagnosis}_{s.skin_tone}" for s in all_samples]

    # Check if stratification is possible (each stratum needs at least 2 samples)
    from collections import Counter
    key_counts = Counter(stratify_keys)
    min_count = min(key_counts.values())

    if min_count < 2:
        # Fall back to diagnosis-only stratification
        stratify_keys = [str(s.diagnosis) for s in all_samples]
        key_counts = Counter(stratify_keys)
        min_count = min(key_counts.values())

        if min_count < 2:
            # No stratification possible
            stratify_keys = None

    val, test = train_test_split(
        all_samples,
        test_size=1 - val_ratio,
        stratify=stratify_keys,
        random_state=random_seed,
    )

    return val, test


def get_split_statistics(
    val: list[DDISample],
    test: list[DDISample],
) -> dict:
    """
    Get statistics for dataset splits.

    Args:
        val: Validation samples
        test: Test samples

    Returns:
        Dictionary with split statistics
    """
    def compute_stats(samples: list[DDISample], name: str) -> dict:
        n_total = len(samples)
        n_malignant = sum(1 for s in samples if s.diagnosis == 1)

        # Group counts
        group_counts = {}
        for s in samples:
            group_counts[s.demographic_group] = group_counts.get(s.demographic_group, 0) + 1

        return {
            "split": name,
            "n_total": n_total,
            "n_malignant": n_malignant,
            "n_benign": n_total - n_malignant,
            "malignant_rate": n_malignant / n_total if n_total > 0 else 0,
            "group_counts": group_counts,
        }

    return {
        "val": compute_stats(val, "val"),
        "test": compute_stats(test, "test"),
    }
