# Frozen reviewed dataset splits

Dataset SHA-256: `c8f76e80e95337bdfd874780bcad16483a2db02a9539f05ab44794b8a2ae15bf`
Seed: 42

| Split | Actual rows | Target rows | Class counts |
|---|---|---|---|
| train | 420 | 420 | {'Bug Report': 70, 'Customer Feedback': 70, 'Feature Request': 70, 'General Inquiry': 90, 'Billing Issue': 50, 'Non-Qualifying': 70} |
| validation | 90 | 90 | {'General Inquiry': 19, 'Non-Qualifying': 15, 'Billing Issue': 11, 'Feature Request': 15, 'Customer Feedback': 15, 'Bug Report': 15} |
| test | 90 | 90 | {'Billing Issue': 15, 'Customer Feedback': 15, 'Bug Report': 15, 'Feature Request': 15, 'General Inquiry': 15, 'Non-Qualifying': 15} |

Count deviations are reported, not hidden. Related scenario groups stay together. All original test-pool IDs remain test-only. Source/review hashes and IDs are retained in the manifest.

Review was performed by the assistant on the user's explicit delegation; no human label review is claimed. Synthetic data remains synthetic after review. No model was trained or evaluated.
