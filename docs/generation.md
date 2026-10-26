# One complete ticket (Stage 7)

OpenAI is the selected provider. The configurable default is `gpt-5.4-mini`; its current official documentation lists Responses and Structured Outputs support. The installed OpenAI Python SDK is **2.54.0**, and the adapter uses `client.responses.parse(..., text_format=GeneratedDetails)`. A real request succeeded with this account/model on 2026-10-26. See the [model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs), and [official SDK setup](https://developers.openai.com/api/docs/libraries).

## Local configuration

Stage 10 fixes the current policy at **ticket-details-v2**. The general guard now rejects first-person future commitments such as promising to pass feedback along, and batch UI results invalidate on prompt-version changes. This was corrected from a known development draft before held-out evaluation; no sample-specific answer rules were added. Actual acceptance errors, retries, priority disagreements and semantic draft concerns are in the [evaluation guide](evaluation.md).

Use the ignored project-root `.env`. It now contains the locally supplied key; `.env.example` has a blank placeholder only. Paste keys into `.env`, never into chat or source control:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
OPENAI_API_KEY=
LLM_TIMEOUT_SECONDS=20
LLM_MAX_RETRIES=1
LLM_MAX_OUTPUT_TOKENS=600
```

Environment variables override `.env`. Keys are excluded from settings representations/serialization, session results, request prompts and reports. Provider errors use safe application messages rather than raw SDK error bodies. Requests use the official OpenAI endpoint, with no alternate provider or heuristic fallback. The key is read when needed; use **Refresh API setup** after editing configuration.

## Use Streamlit

Open `http://127.0.0.1:8501` after starting the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The separate **Complete analysis of one ticket** form accepts an ID and the original customer message. Click **Analyze one ticket** after both local checkpoints and API configuration are ready. The original message and local category/sentiment remain visible. Category and sentiment truncation are flagged independently (128 and 512 tokens respectively); the API receives the preserved original message, limited to 12,000 characters by this stage's input contract.

Blank/exact greeting-only messages are ignored before model prediction. A local Non-Qualifying prediction skips sentiment and the API. Qualifying tickets receive local category and sentiment predictions, followed by one structured generation operation. Only a valid complete result is `processed`; failed processing remains `error`, with available local results retained. Replies are unsent drafts.

The existing CSV **Analyze categories** and **Analyze sentiment** buttons still produce partial local analysis. [Stage 8 Analyze tickets](batch.md) runs the complete shared pipeline and persists results in SQLite. The single-ticket form described here keeps its session-only behavior. Exports belong to Stage 9.

Session results are keyed by ticket ID/message, safe configuration, prompt version, model-file stamps and local configuration-file stamp. Reruns and resubmitting an already analyzed input reuse its outcome, including when returning to an earlier input. To deliberately retry/reanalyze, select the explicit reprocessing checkbox and submit; this may incur another API charge. This is per-session memory, not durable storage or a billing guarantee.

## Output and policies

The API may return exactly three required fields:

| Field | Local validation |
|---|---|
| `priority` | `Low`, `Medium`, `High` or `Urgent` |
| `product_feature` | A phrase present in the original message, or `null`; case/whitespace differences are accepted |
| `suggested_reply` | Nonblank, at most 600 characters and one or two sentences |

Category and sentiment are supplied as authoritative local results; the API cannot overwrite them. Priority follows reported impact: widespread outage, security/payment-blocking situations or critical loss justify Urgent; serious impairment justifies High; ordinary bugs/requests generally use Medium; low-impact informational feedback can use Low. Sentiment and words such as “ASAP” alone do not establish urgency. There are no fixture-specific answers.

The system instructions treat customer text as untrusted data. The message is JSON-encoded separately from policy, and no tools or external actions are enabled. Product extraction must be grounded in the message. Drafts should acknowledge the issue and offer a suitable next question/step, without claiming a refund, fix, escalation or other action occurred or promising a resolution time.

Pydantic rejects unexpected fields, invalid labels, ungrounded product phrases and common action/promise violations. These conservative lexical checks can reject legitimate wording and cannot prove every semantic claim; a person must still review the priority and reply. Invalid output/refusals remain errors; there is no automatic repair call or invented replacement reply.

## Timeouts and retries

Inspection of the installed SDK found a default of two retries and a 600-second read timeout. The application explicitly sets **SDK retries to zero**, a **20-second I/O timeout** (connect at most five seconds), and at most **one application retry**. Only a transient HTTP 429 (excluding insufficient quota) or server failure is eligible, with a retry delay of at most two seconds. A longer requested delay is returned as an error.

Authentication/quota failures, refusals, incomplete responses and invalid schema receive no retry. Timeouts and connection errors also receive no automatic retry because request completion/billing may be uncertain. Normally a ticket makes one request; an eligible transient failure can make two physical attempts. I/O limits are not an exact total wall-clock or exactly-once billing guarantee.

## Configuration and real integration checks

```powershell
# Configuration only: no local inference or API request.
.\.venv\Scripts\python.exe -m scripts.check_ticket
# Explicit live check: real CPU models and OpenAI on sample T001.
.\.venv\Scripts\python.exe -m scripts.check_ticket --live
```

`--input`, `--ticket-id` and `--output` allow another development fixture/report path. Do not use held-out test data for this integration check. A successful live run records actual local predictions, validated API fields, requested model, response ID, attempts, token usage, duration and SDK version in [live_check.json](../reports/single_ticket/live_check.json), and prints its actual reply draft. No reply is sent. A configuration-only run writes [configuration_check.json](../reports/single_ticket/configuration_check.json), preserving a previous live report.

Click **Show saved live integration check** in Streamlit to render a successfully validated real report without another request. Pending configuration does not display a fabricated reply, and mocked test results are not saved as live reports.

## Recorded verification — 2026-10-26

The initial missing-key check made no API request; its real Streamlit AppTest confirmed that the complete action was disabled and no fake live reply appeared. That historical guard is recorded in [ui_configuration_check.json](../reports/single_ticket/ui_configuration_check.json).

After local credentials were supplied, a real `--live` run processed sample **T001** using the saved CPU models and OpenAI. Actual fields were **Bug Report / Negative / Urgent**, with product/feature **Checkout**. The request succeeded on **one attempt in 6.317 seconds**, using 471 input and 58 output tokens. This is one measured development integration check, not an accuracy or general performance estimate. [Actual result and receipt](../reports/single_ticket/live_check.json)

The actual unsent reply was:

> Thanks for reporting this. I am sorry the payment page is crashing on Chrome when you click Checkout; could you share any error message or the browser version you are using?

A separate Streamlit AppTest displayed that saved real reply, its draft notice and complete fields; viewing it and rerunning made **zero additional API calls**. The complete action was enabled with configuration. See [streamlit_live_check.json](../reports/single_ticket/streamlit_live_check.json). This exercised Streamlit's actual rendering logic; it was not a visual browser check. The local server runs at `http://127.0.0.1:8501`; click **Show saved live integration check** to view the recorded draft without another request.

The full suite passed **318 tests in 90.62 seconds**. Stage 7 adds 37 generation/schema/adapter tests, 11 orchestration tests, 10 Streamlit integration tests and two CLI guards. These use explicit mocks, including the installed SDK parser through `httpx.MockTransport`; they verify schema/refusal/failure handling, local label authority, safe errors, qualification boundaries and rerun/reprocessing behavior. Mock transport verifies request/parse integration without contacting OpenAI. [sdk_support.json](../reports/single_ticket/sdk_support.json) records the installed SDK/defaults and official support references. No held-out intent-test content was inspected.
