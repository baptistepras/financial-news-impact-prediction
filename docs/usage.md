# Usage

Setup, then every command with its options and outputs. How each stage works is in [implementation.md](implementation.md).

## Setup

The project has its own environment, `financial-news`, defined in [`environment.yml`](../environment.yml).

```bash
mamba env create -f environment.yml          # create the environment once
mamba activate financial-news                # activate it in every new terminal
mamba env update -f environment.yml --prune  # after environment.yml changes
```

The cleaning steps need the fastText language identification model at the project root:

```bash
curl -L -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Every command below runs from the project root.

## Commands

| Command | What it does | Outputs |
| --- | --- | --- |
| `python summarization/train_chunk_reduce.py` | Cleans `data/dataset.csv`, fine-tunes Flan-T5-large, and generates the test summaries with the map-reduce strategy. Needs a CUDA GPU. | Checkpoints in `t5_impact_large/`, then `results/pred.csv`. |
| `python summarization/evaluate.py` | Grades every summary of `results/pred.csv` with Qwen2.5-7B-Instruct. Needs about 16 GB of GPU memory in float16. | `results/judge_results.json` |
| `jupyter lab entity_linking.ipynb` | Dataset audit, NER, and ticker linking, cell by cell. Queries Yahoo Finance. | Event tables in the notebook. |
| `python market_reaction.py --train_csv data/train_triplets.csv --test_csv data/test_triplets.csv --output_dir results/market_reaction --index_ticker "^TASI.SR"` | Downloads daily prices, labels each event, and trains and evaluates the classifier. | Labeled events, `metrics.json`, and plots in `results/market_reaction/`. |

## Options

**`summarization/train_chunk_reduce.py`**: `--data`, `--lid`, `--impact_model_name` and `--notes_model_name` (`google/flan-t5-large`), `--output_dir`, `--pred_csv`, `--test_size` (0.20), `--chunk_len` (768), `--stride` (384), `--lr` (8e-5), `--epochs` (6), `--train_bs` (2), `--grad_accum` (8), `--first_chunk_weight` (1.35), `--num_beams` (6), `--length_penalty` (1.2), `--reduce_group_size` (10), `--fp16`.

**`summarization/evaluate.py`**: `--input_csv`, `--out_json`, `--model` (`Qwen/Qwen2.5-7B-Instruct`), `--max_rows` (0 for all), `--torch_dtype` (`float16`, `bfloat16`, or `float32`), `--device_map` (`auto`), `--max_source_chars` (9000).

**`market_reaction.py`**: `--train_csv` and `--test_csv` (required), `--output_dir`, `--index_ticker` (`^TASI.SR`), `--ticker_suffix`, `--embed_model` (`all-MiniLM-L6-v2`), `--epochs` (15), `--lr` (2e-3), `--batch_size` (64), `--val_frac` (0.15), `--seed` (42).
