"""
TruthLens AI - Master Dataset Generation & Stratification Script
Executes Phase 2 dataset generation, schema validation, and split export.
"""

import json
from pathlib import Path
import pandas as pd
from src.data.dataset_builder import build_benchmark_dataset, stratify_splits
from src.data.schema import MultimodalSampleMetadata

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("TruthLens AI: Phase 2 - Dataset Construction & Stratification")
    print("=" * 60)

    # 1. Build samples
    print("\n[1/4] Generating 450 benchmark samples...")
    samples = build_benchmark_dataset(
        output_dir=RAW_DIR,
        total_real=200,
        total_fake=200,
        total_uncertain=50
    )
    print(f"-> Successfully generated {len(samples)} multimodal samples.")

    # 2. Stratify splits (70/15/15)
    print("\n[2/4] Performing stratified 70/15/15 split...")
    splits = stratify_splits(samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    for split_name, split_samples in splits.items():
        print(f"   * {split_name.upper()} split: {len(split_samples)} samples")

    # 3. Export to CSV & JSON
    print("\n[3/4] Exporting processed metadata and split files...")
    all_dicts = [s.model_dump() for s in samples]
    
    # Save master datasets
    with open(PROCESSED_DIR / "dataset_master.json", "w", encoding="utf-8") as f:
        json.dump(all_dicts, f, indent=2)
    df_master = pd.DataFrame(all_dicts)
    df_master.to_csv(PROCESSED_DIR / "dataset_master.csv", index=False)

    # Save individual split CSV & JSON
    for split_name, split_samples in splits.items():
        split_dicts = [s.model_dump() for s in split_samples]
        with open(SPLITS_DIR / f"{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(split_dicts, f, indent=2)
        pd.DataFrame(split_dicts).to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)

    # 4. Generate statistics summary
    print("\n[4/4] Dataset Validation & Class Distribution:")
    print("-" * 50)
    summary = df_master.groupby(["split", "pair_label"]).size().unstack(fill_value=0)
    print(summary)
    print("-" * 50)
    print("Manipulation Type Distribution:")
    print(df_master["manipulation_type"].value_counts())
    print("-" * 50)
    print(f"Phase 2 complete! Dataset saved under {DATA_ROOT.resolve()}")


if __name__ == "__main__":
    main()
