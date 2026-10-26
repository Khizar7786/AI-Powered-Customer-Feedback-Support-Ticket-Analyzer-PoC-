# Visible acceptance fixtures

`assessment.csv` preserves the three exact supplied examples. `mixed.csv` is a separate assistant-authored engineering batch covering all six categories, greeting-prefixed/short/URL issues, untrusted embedded instructions, a promotion, exact greeting, missing/duplicate IDs, malformed fields and a 14,640-character input-limit case.

`expectations.json` was written before inference. Assessment expectations come from the assessment; mixed expectations are the assistant's interpretation of the documented policy. They are references for reporting disagreements, not fields injected into model/API input. They are not independently human-reviewed labels or a sentiment-accuracy dataset. No training/validation/held-out examples were used to author this batch.

Run `python -m scripts.check_acceptance`. Matching stored outcomes are reused; `--retry-errors` explicitly retries previous errors and can make another paid request. Original runs/exports remain under `reports/acceptance/<batch>/runs/`. Actual mismatches and draft concerns are in `reports/acceptance/review.md`. Never force model outputs to the reference values.
