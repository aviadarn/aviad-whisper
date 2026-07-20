# aviad-whisper

Fine-tune OpenAI Whisper on Apple Silicon (M5 / MPS) to help you write faster by dictation.
End-to-end flow: **audio → split → batch → train → validate (WER + F1)**.

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run the whole pipeline

```bash
python -m src.pipeline
```

Prints WER + word-level F1 on the held-out test split and saves a fine-tuned
checkpoint to `models/whisper-finetuned/`, metrics to `models/test_metrics.json`.

## Run a stage alone

```bash
python -m src.data          # load + show splits
python -m src.pipeline --skip-train   # eval an existing checkpoint
```

## Pipeline stages (`src/`)

| file               | job                                                  |
|--------------------|------------------------------------------------------|
| `data.py`          | load dataset, split train/val/test                   |
| `preprocess.py`    | audio→log-mel features, text→tokens, batching collator |
| `train.py`         | Seq2SeqTrainer fine-tune on MPS                      |
| `metrics.py`       | WER (jiwer) + word-level F1                          |
| `evaluate_model.py`| generate on test split, score                       |
| `pipeline.py`      | chain 1→6                                            |

## Collect your own voice (`record/`)

Record yourself reading generated prompts to build a personal dataset.

```bash
python record/app.py            # open http://localhost:8000
```

Browser shows one sentence at a time. Read it, hit **Record** (or `Space`),
**Stop**, then **Save & Next** (`Enter`). Each clip is transcoded to 16 kHz mono
WAV and paired with its exact text in `data/voice/metadata.csv`
(`audiofolder` format). Progress is saved; re-open any time to resume.

Then train on your voice — flip one line in `config.yaml`:

```yaml
dataset:
  source: local     # was: hub
```

and run `python -m src.pipeline`. Everything else is unchanged. Aim for
100+ clips before expecting useful accuracy.

## Config

Everything is in `config.yaml`. `dataset.source: hub` uses the public set;
`local` uses your own recordings in `data/voice`.

## Notes

- **WER** is the standard ASR metric (lower = better). **F1** is word-level
  (precision/recall over the word multiset), included per request.
- First run uses `PolyAI/minds14` (tiny) and `max_steps: 200` to get the loop
  green fast. Raise `max_steps` / swap the model in `config.yaml` for real training.
