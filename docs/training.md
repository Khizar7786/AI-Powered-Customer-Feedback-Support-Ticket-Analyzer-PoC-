# Train the category classifier (Stage 4)

Run from the project root with the editable package installed. Training is a separate terminal workflow. The saved model now serves the complete [batch pipeline](batch.md). Stage 10 evaluated this fixed checkpoint once: 98.89% held-out category accuracy / 0.9889 macro-F1 on the synthetic assistant-reviewed corpus. See the [evaluation guide](evaluation.md); do not retrain/tune against this now-consumed test set while describing it as blind.

```powershell
# Check the reviewed/frozen data and available device first.
.\.venv\Scripts\python.exe -m scripts.check_environment
# Small real run: 12 training / 6 validation rows, one epoch, separate artifacts.
.\.venv\Scripts\python.exe -m scripts.train_intent --smoke
.\.venv\Scripts\python.exe -m scripts.train_intent --smoke --verify-only
# Full reproduction from the included data/splits/{train,validation}.csv and manifest.
# Use a new report directory because the original measured report is included.
.\.venv\Scripts\python.exe -m scripts.train_intent --revision 12040accade4e8a0f71eabdb258fecc2e7e948be --report-dir reports/intent_training_reproduction
# Run this as a separate command/process after training.
.\.venv\Scripts\python.exe -m scripts.train_intent --verify-only --report-dir reports/intent_training_reproduction
```

The first training invocation downloads the public base model to `.cache/huggingface/hub`. To reproduce the recorded experiment, pass `--revision 12040accade4e8a0f71eabdb258fecc2e7e948be`; add `--local-files-only` after that revision is cached. The full run starts from the pretrained encoder again; smoke weights are never reused. `--verify-only` loads the saved tokenizer and model locally on CPU and checks the label mappings and one finite six-score forward pass. It reads no dataset file and makes no API request.

## Data boundary

Complete the [review/freezing workflow](data-review.md) first. The command requires a valid frozen manifest, matching train/validation hashes and membership, reviewed development rows, all six categories in both splits, unique IDs/messages, and separate scenario groups. Changing a CSV requires a newly reviewed/frozen dataset version. The loader opens only `train.csv`, `validation.csv`, and manifest metadata. It never opens `test.csv`, the raw draft, or the combined reviewed corpus. Held-out evaluation belongs to Stage 10.

The current corpus was reviewed by the assistant under explicit user delegation. Its provenance states `human_review_performed: false`; all messages are synthetic. The recorded validation scores measure this small corpus and do not establish real-customer accuracy. Class balancing retained related groups; the actual training/validation counts are:

| Category | Training | Validation |
|---|---:|---:|
| Bug Report | 70 | 15 |
| Feature Request | 70 | 15 |
| Billing Issue | 50 | 11 |
| General Inquiry | 90 | 19 |
| Customer Feedback | 70 | 15 |
| Non-Qualifying | 70 | 15 |

Only original `message` strings enter the tokenizer. `label` becomes the numeric target. IDs, provenance, review status, pools and scenario groups never become model features. Original CSV text is unchanged. Reports count messages truncated at 128 tokens.

## Settings and memory

The fixed base model is `distilbert/distilbert-base-uncased`, fine-tuned for all six labels. Defaults are 3 epochs, learning rate `2e-5`, torch AdamW, weight decay `0.01`, maximum length 128, train/evaluation batch size 2, accumulation 4, seed 42, linear scheduling with 10 warmup optimizer steps, zero Windows loader workers, and no external experiment tracking. Smoke uses one epoch and zero warmup. Evaluation and saving happen each epoch; Trainer reloads the checkpoint with the highest validation macro-F1 before exporting the final model. Two epoch checkpoints are retained.

Unload sentiment and other application GPU models before training. The command checks actual free CUDA memory and float32/float16 arithmetic. FP16 requires supported CUDA hardware and a passing check; CPU uses full precision. Dynamic padding adds only enough padding for the current batch (rounded to a multiple of eight for FP16).

Below 2,500 MiB free, the default settings switch to train batch 1, evaluation batch 1, accumulation 8. Below 1,500 MiB, the command stops with guidance to free GPU memory or choose CPU. An unexpected CUDA out-of-memory failure is recorded; retry explicitly with new paths:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_intent --batch-size 1 --eval-batch-size 1 --gradient-accumulation 8 --output-dir models/ticket_classifier_retry --report-dir reports/intent_training_retry
# CPU training alternative (slower; same split and model):
.\.venv\Scripts\python.exe -m scripts.train_intent --device cpu --output-dir models/ticket_classifier_cpu --report-dir reports/intent_training_cpu
```

Nonempty output/report directories are protected from overwrite. For a deliberate new experiment, choose new paths with `--output-dir` and `--report-dir`; do not delete the selected model by accident. CPU reload was exercised; a full CPU training run has not been benchmarked.

## Saved artifacts

The best checkpoint is exported to `models/ticket_classifier/`, including `model.safetensors`, `config.json` with `label2id`/`id2label`, the tokenizer/vocabulary, separate `label_mappings.json`, training summary, Trainer arguments and development-data manifest. Epoch checkpoints remain under `checkpoints/`. Smoke outputs go to `models/intent_smoke/` and `reports/intent_smoke/`.

`reports/intent_training/` contains actual configuration and library versions, dataset and split fingerprints, measured training/pipeline duration, peak GPU memory, epoch/step history, validation loss/accuracy/macro-F1, per-class precision/recall/F1, a confusion matrix, validation prediction IDs, run status and the separate-process reload check. Smoke metrics are labeled smoke and have no accuracy significance. Failed runs retain an explicit failed status.

## Recorded local run — 2026-10-26

The smoke check completed on CUDA in **3.36 seconds** (training/evaluation/checkpoint interval), changed the classifier head and reloaded in a fresh CPU process. The full run then started from the base model at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`, with the default settings unchanged:

| Epoch | Validation loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 1 | 1.2713 | 0.7444 | 0.7269 |
| 2 | 0.6726 | 0.8889 | 0.8824 |
| 3 (selected) | 0.5227 | 0.9333 | 0.9305 |

Full training, epoch validation/checkpoint saving and best-model loading took **43.44 seconds**; the pipeline through model export took **56.41 seconds**, excluding the earlier download. There were 159 optimizer steps. CUDA and FP16 arithmetic passed on the RTX 3050 Ti Laptop GPU (4,095.5 MiB total; 3,239.7 MiB free after the CUDA probe). Peak PyTorch allocation was 1,299.68 MiB, with 1,412 MiB reserved. No low-memory fallback was needed. Other GPU model processes were absent before the run. None of the development messages exceeded 128 tokens (longest train/validation lengths: 46/52).

The selected export's weights exactly match `checkpoint-159/model.safetensors` by SHA-256. A separate CPU process loaded the final tokenizer/weights, verified all six saved label mappings and produced finite logits of shape `[1, 6]`. See [summary](../reports/intent_training/training_summary.json), [validation metrics](../reports/intent_training/validation_metrics.json), [reload check](../reports/intent_training/reload_check.json), [artifact checks](../reports/intent_training/artifact_checks.json) and [measured curves](../reports/intent_training/training_curves.png). Non-Qualifying had validation recall 0.7333 (11/15), an observed limitation even in this synthetic corpus. No extra epochs or held-out evaluation were run.

Dataset fingerprint: `c8f76e80e95337bdfd874780bcad16483a2db02a9539f05ab44794b8a2ae15bf`. Runtime: Python 3.12.5, torch 2.11.0+cu128, Transformers 4.57.6, Datasets 4.8.5, Accelerate 1.15.0, scikit-learn 1.9.1, Pydantic 2.14.1. The full suite passes 127 tests, including 25 new offline Stage 4 contract tests. Those tests are separate from the actual smoke/training/reload checks.

A separate [Streamlit AppTest check](../reports/intent_training/ui_regression_check.json) used the actual saved checkpoint's file presence, loaded the sample preview and reran successfully: 12 rows, 6 candidates, 2 ignored and 4 input errors. Analysis stayed disabled and no model weights were loaded by the UI. This was a simulated Streamlit session, not a browser visual check.

Weights and caches are ignored by Git. To hand off a trained model, transfer the entire `models/ticket_classifier/` directory through an agreed artifact channel; the root model/tokenizer/metadata files suffice for reload, while `checkpoints/` can be omitted to save space. Place it at that path on the receiving machine, install dependencies, then run `--verify-only`. No external artifact upload has been performed. Alternatively transfer the reviewed development split files and manifest and reproduce training with the pinned revision. Keep the held-out file separately for final evaluation.

## Explain the training in an interview

- **Tokenization:** The tokenizer breaks a message into vocabulary pieces and converts them to numbers. An attention mask tells the model which positions contain text and which are padding. Messages longer than 128 tokens are truncated for this experiment.
- **Logits and loss:** The model produces six raw scores called logits. Cross-entropy loss measures how poorly those scores support the correct category; a lower loss indicates a better fit to those targets. The classification head starts with new weights and must be trained.
- **Backpropagation and accumulation:** Backpropagation computes how each weight contributed to the loss. With batch 2 and accumulation 4, gradients from four small batches are combined before AdamW updates the weights: normally eight messages per update, with a smaller remainder at the end of an epoch. This saves GPU memory.
- **Checkpoint selection:** Training rows teach the weights. Validation rows are checked after each epoch without updating them. Macro-F1 gives each of the six categories equal weight, and its best epoch is saved as the final model. Held-out test messages remain unused until model choices are fixed.

Implementation references: [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) and [Trainer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer). The actual experiment records the installed library versions rather than assuming the documentation version is the runtime version.
