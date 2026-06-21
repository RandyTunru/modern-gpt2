# modern-gpt2

A from-scratch ~110M-parameter decoder-only language model and its full training pipeline: **pretraining → continued pretraining with context extension → supervised fine-tuning (SFT)**. The architecture borrows modern LLaMA/Mistral-style components (RoPE, RMSNorm, weight tying) and uses the Mistral tokenizer.

## Pipeline at a glance

| Phase | Config | Data | Seq len | Starts from |
|-------|--------|------|---------|-------------|
| **1. Pretrain** | `configs/train_gpt2_small.yaml` | Cosmopedia | 1024 | scratch |
| **2. Continued pretrain (CPT)** | `configs/cpt_2048_small.yaml` | Cosmopedia (long docs) | 2048 | pretrain checkpoint |
| **3. SFT** | `configs/sft_2048_small.yaml` | OpenHermes-2.5 | 2048 | CPT checkpoint |

Context extension (1024 → 2048) is done by **RoPE frequency scaling**: the model is built with `original_seq_len=1024` and `max_seq_len=2048`, so the rotary frequencies are rescaled to the new window during CPT ([`src/models/gpt.py`](src/models/gpt.py)).

## Architecture

Defined in [`src/models/gpt.py`](src/models/gpt.py) and [`src/models/components.py`](src/models/components.py):

- Decoder-only transformer, 12 layers / 12 heads / `d_model=768` / `d_ff=2048` (~110M params)
- **RoPE** rotary positional embeddings (with context-extension scaling)
- **RMSNorm** pre-normalization
- **Weight tying** between the token embedding and the output projection
- Attention via `F.scaled_dot_product_attention`, with an optional **block-diagonal mask** for packed SFT sequences
- `vocab_size=32000` (Mistral-7B-v0.1 tokenizer)

## Repository layout

```
configs/                 # YAML training configs, one per phase
scripts/
  cosmopedia_prep.py     # Tokenize Cosmopedia into a flat token stream (pretrain/CPT)
  openhermes_prep.py     # Tokenize + smart-pack OpenHermes into bins (SFT)
  train.py               # Single entry point for all training phases
  generate.py            # Inference / interactive chat
src/
  data/dataset.py        # PretrainDataset + SFTDataset
  models/gpt.py          # GPT model + generate()
  models/components.py   # Attention, FFN, RMSNorm, Block
  training/trainer.py    # Training loop (grad accumulation, shifting, masking)
  utils/metrics.py       # Param count, throughput monitor
artifacts/checkpoints/   # Saved checkpoints, per phase
```

## 1. Data preparation

### Pretraining / CPT data (Cosmopedia)

Produces a single flat `uint16` token stream consumed by `PretrainDataset`.

```bash
# Pretrain corpus
python scripts/cosmopedia_prep.py \
  --subset stanford \
  --output-file data/processed/train_data.bin

# CPT corpus (keep only longer documents for context extension)
python scripts/cosmopedia_prep.py \
  --subset web_samples_v1 \
  --min-tokens 1024 \
  --output-file data/processed/web_v1_min_1024_cosmopedia.bin
```

### SFT data (OpenHermes, smart-packed)

Conversations are formatted with the Mistral template, then **bin-packed** to a fixed `max_seq_len` so little of each sequence is wasted. The prep emits **four** files (the `.bin` plus three sidecars):

| File | dtype | Contents |
|------|-------|----------|
| `<name>.bin` | uint16 | token ids |
| `<name>_labels.bin` | int32 | targets (`-100` masks BOS / user turns / padding) |
| `<name>_doc_ids.bin` | int16 | per-bin document id (`0` = padding); drives the block-diagonal mask |
| `<name>_meta.json` | - | `max_seq_len`, dtypes, bin count |

```bash
python scripts/openhermes_prep.py \
  --max-seq-len 2048 \
  --output-file data/processed/openhermes/2048_openhermes.bin
```

> **Important:** packed data is bound to the `max_seq_len` it was generated with. The doc-ids are *bin-relative*, so reading at a different sequence length would straddle bins and let unrelated conversations attend to each other. `SFTDataset` reads `_meta.json` and **hard-fails** if the training `max_seq_len` doesn't match the packed one.

## 2. Training

All three phases use the same entry point; the config decides everything:

```bash
# Phase 1: pretrain (1024)
python scripts/train.py --config configs/train_gpt2_small.yaml

# Phase 2: continued pretrain / context extension (2048)
python scripts/train.py --config configs/cpt_2048_small.yaml

# Phase 3: SFT (2048, packed + masked)
python scripts/train.py --config configs/sft_2048_small.yaml
```

Notable config keys:

- `from_checkpoint`: load weights from a prior phase (RoPE/mask buffers are dropped and rebuilt for the new `max_seq_len`).
- `is_resume`: if `true`, also restore optimizer state and step count (true resume vs. starting a new phase).
- `use_sft_masking`: switches the loader to `SFTDataset` and enables block-diagonal attention from `doc_ids`.
- `original_seq_len`: the pretraining context length, used for RoPE scaling during CPT/SFT.
- `no_epochs`: train only by `max_steps` rather than epochs (dataset will not be repeated).

Labels are shifted **in the training loop**, not in the dataset (`logits[:-1]` vs `labels[1:]`), with `ignore_index=-100` so the same loss path serves both pretraining (no masked tokens) and SFT (masked user/pad tokens). See [`src/training/trainer.py`](src/training/trainer.py).

Training logs to Weights & Biases (project/run names come from the config). Checkpoints are written to `checkpoint_dir` every `save_every` steps, plus a final `model_step_final.pt`.

## 3. Inference

```bash
# Single completion from a base/pretrained checkpoint
python scripts/generate.py "The history of Rome" \
  --config configs/cpt_2048_small.yaml

# Single chat turn from the SFT model (wraps the prompt in the chat template)
python scripts/generate.py "Explain RoPE simply." \
  --config configs/sft_2048_small.yaml \
  --apply-chat-template

# Interactive multi-turn chat
python scripts/generate.py "Hi!" \
  --config configs/sft_2048_small.yaml \
  --multiturn
```

Useful sampling flags: `--max-new-tokens`, `--temperature`, `--top-k`, `--repetition-penalty`.

> When evaluating the SFT model, always pass `--apply-chat-template` (or `--multiturn`). Without it the prompt is fed raw, which is the *pretraining* distribution and the model behaves like a plain completer rather than a chat assistant.

## Configs

| Config | Phase | Seq len | LR | Steps |
|--------|-------|---------|----|-------|
| `train_gpt2_small.yaml` | Pretrain | 1024 | 6e-4 | 50000 |
| `cpt_2048_small.yaml` | CPT | 2048 | 2e-5 | 10000 |
| `cpt_4096_small.yaml` | CPT (4096) | 4096 | — | — |
| `sft_2048_small.yaml` | SFT | 2048 | 2e-5 | 5000 |
