# Final held-out intent evaluation

Status: complete
Rows: 90; accuracy: 0.9889; macro-F1: 0.9889.
Coverage: 1.0000; duration: 12.58 seconds.

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Bug Report | 1.0000 | 1.0000 | 1.0000 | 15.0 |
| Feature Request | 1.0000 | 0.9333 | 0.9655 | 15.0 |
| Billing Issue | 1.0000 | 1.0000 | 1.0000 | 15.0 |
| General Inquiry | 0.9375 | 1.0000 | 0.9677 | 15.0 |
| Customer Feedback | 1.0000 | 1.0000 | 1.0000 | 15.0 |
| Non-Qualifying | 1.0000 | 1.0000 | 1.0000 | 15.0 |

Valid tickets incorrectly ignored: 0/75.
Non-qualifying tickets incorrectly accepted: 0.

Small synthetic English corpus; review was assistant-performed under explicit user delegation where recorded, not independently human-validated. Held out from training/model selection since freezing, but assistant-authored/reviewed scenarios are not guaranteed independent real-customer data. These category metrics do not measure sentiment, priority, reply quality or end-to-end application accuracy.

No model/prompt tuning, training or API calls occurred during this category evaluation. The held-out set is now evaluated; a future tuning cycle needs a new independent test set.
