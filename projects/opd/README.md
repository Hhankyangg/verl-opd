# OPD Data

This directory contains portable data-preparation helpers for OPD experiments.

Prepare the MINT-CoT parquet used by the BAGEL OPD scripts:

```bash
PYTHON_BIN=/path/to/verl-env/bin/python bash projects/opd/prepare_data/run_mint_cot_prep.sh
```

Expected outputs:

```text
projects/opd/data_parquet/mint_cot_dataset/train.parquet
projects/opd/data_parquet/mint_cot_dataset/val_sample100.parquet
```

If the raw MINT-CoT parquet is already available, set `RAW_MINT_FILE`:

```bash
RAW_MINT_FILE=/path/to/MINT-CoT_interleave_rl_54k_filtered.parquet \
PYTHON_BIN=/path/to/verl-env/bin/python \
bash projects/opd/prepare_data/run_mint_cot_prep.sh
```

The launch scripts read `DATA_DIR`, `TRAIN_DATA_FILE`, and `VAL_DATA_FILE`, so
you can also point them to externally prepared parquet files.
