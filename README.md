# Customer Support Copilot

A responsive workspace for support agents to find policy-backed product guidance, review citations, and edit a customer reply before sending it themselves. The app never sends a reply.

## Stack choice

The starting workspace was empty apart from the earlier RAG demo. This version keeps the useful local-first approach: Python's standard library serves the API, while plain HTML/CSS/JavaScript provides a responsive interface. It needs no package install, database service, or build step, so it is easy to run and inspect. The tradeoff is that its in-memory index and demo login are for a trusted local environment, not a production support platform.

## Run locally

Requires Python 3.10+. The app uses only the Python standard library; `requirements.txt` documents that no packages need to be installed.

```bash
python3 server.py
```

Open <http://127.0.0.1:8000>. In local demo mode the server uses the `agent` role and does not require login. Sample sources are in `knowledge/`; uploads are stored there too. Supported source formats are JSON, Markdown, and plain text, up to 500 KB each. JSON sources can include metadata fields such as `title`, `type`, `product`, `plan`, `region`, `issue_type`, `updated`, `effective_date`, `authority`, and `roles`, plus required `content`. Markdown supports simple front matter.

## Deploy to Render

The repository includes a Render Blueprint in `render.yaml`.

1. Push this project to a GitHub repository.
2. In Render, choose **New → Blueprint**, connect the repository, and let Render read `render.yaml`.
3. When prompted for `SUPPORT_USERS`, enter a JSON object with at least one account. Use a unique, long password and keep it in Render’s environment settings; do not commit it. Example shape:

   ```json
   {"agent":{"password":"use-a-long-unique-secret","role":"agent"},"specialist":{"password":"another-long-unique-secret","role":"specialist"},"admin":{"password":"a-third-long-unique-secret","role":"admin"}}
   ```

4. Deploy, then open the service URL and sign in. Render checks `/healthz` before routing traffic. `APP_ENV=production` makes the app refuse to start if `SUPPORT_USERS` is missing and marks the session cookie `Secure`.
5. Add `OPENAI_API_KEY` and `OPENAI_MODEL` only if you want provider-generated drafts. Add `EMBEDDING_MODEL` and its optional provider key/base URL only if you want semantic retrieval. Keep all credentials in the host’s secret environment settings.

The Blueprint uses Render’s Python runtime, `pip install -r requirements.txt`, and `python3 server.py`; the server binds to the host and port supplied by the platform. See Render’s [Blueprint reference](https://render.com/docs/blueprint-spec) and [web service setup](https://render.com/docs/web-services) for current dashboard steps.

### Before using real customer data

This is a deployable demo, not a hardened multi-tenant support system. The built-in login is a small environment-configured account list; sessions live in process memory and disappear on restart. Keep it to one instance unless you replace sessions with a shared store. The included knowledge folder ships with the deployment, but uploaded files on a typical ephemeral host can disappear during a restart or redeploy; use durable storage before relying on uploads. Add your identity provider, shared session storage, persistent knowledge storage, rate limiting, audit logging, and a security review before handling real customer data. The app does not send replies.

The generated `outputs/customer-support-copilot.zip` is a source package. Extract it, create a GitHub repository, then push the extracted files (or push this project directly). Do not commit `.env` files, provider keys, real customer records, or local test/runtime caches.

## Optional sign-in and roles

Set `SUPPORT_USERS` to a JSON object in the server environment. Passwords stay server-side and are never sent to browser code. Example for local testing:

```bash
export SUPPORT_USERS='{"agent":{"password":"change-this","role":"agent"},"specialist":{"password":"change-this-too","role":"specialist"},"admin":{"password":"another-secret","role":"admin"}}'
python3 server.py
```

Source visibility comes from each document's `roles` metadata. Agent, specialist, and admin are the supported roles. Uploading sources requires the specialist or admin role, and removal is admin-only. Failed sign-ins are rate limited (5 attempts per 5 minutes per IP). For a real deployment, connect this server-side role check to the company's identity provider, use HTTPS, and store credentials in a secret manager.

## Optional model and embeddings

With no model credentials, retrieval stays local. Reviewed, citation-backed templates support the UPI payment-status, Airtel Prepaid receipt, and Swiggy order-help examples. For other questions, the app shows evidence and asks the agent to write from it or escalate instead of guessing. To synthesize answers, configure an OpenAI-compatible chat completions provider:

```bash
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-chat-model'
# Optional; defaults to https://api.openai.com/v1
export OPENAI_BASE_URL='https://api.openai.com/v1'
```

Optional embeddings use an OpenAI-compatible embeddings endpoint. Set `EMBEDDING_MODEL`, with `EMBEDDING_API_KEY` and `EMBEDDING_BASE_URL` if it differs from the chat provider. Without embedding configuration, retrieval uses local BM25 keyword ranking. The `VECTOR_STORE` setting accepts `memory` (default) or `local` (same local in-memory implementation). `RERANKER_PROVIDER` accepts `hybrid` (default) or `keyword`.

The app displays when external providers are configured. Before model requests it redacts common email, phone, and long-number patterns from the question and evidence. Request bodies and credentials are not logged; only method, route, and status are logged. Data sent to a configured provider is still subject to that provider's terms and retention, so review your organization's data handling requirements before using customer content.

## India-focused sample sources

The sample picker includes everyday UPI payment-status, Airtel prepaid recharge receipt, and Swiggy order-support scenarios. These local records summarize linked official public help pages. Their `updated` date means the demo source was checked on that date; it is not a claim that the provider revised its policy then. Open the source link in the evidence panel to verify current guidance. The examples avoid transaction-specific guarantees and route account or order issues to the provider's official app or bank channel.

## Architecture

`pipeline.py` keeps the RAG stages separate:

1. **Ingest and parse** supported documents and extract metadata.
2. **Chunk** source text into overlapping passages while retaining document identity and metadata.
3. **Filter** by role, product, plan, region, issue type, source kind, and effective date.
4. **Retrieve** with BM25 and, when configured, provider embeddings.
5. **Rerank** with relevance plus source authority and freshness. Current policies and official product documentation outrank resolved tickets; tickets are examples only.
6. **Draft** a concise, editable reply with separate verified facts and suggested steps when a model is configured. Citation IDs are checked against retrieved passages; factual claims need an authoritative source. If evidence is missing, return a brief acknowledgment and specialist escalation.
7. **Inspect** evidence, updated dates, stale guidance, and conflicts before the agent copies or edits the reply.

The reference UI does not provide a send action. It includes copy-to-clipboard for a human to use in their support system.

## Evaluation and checks

Run the deterministic evaluation suite:

```bash
python3 -m unittest discover -s evaluations -v
python3 evaluations/run.py
```

The suite covers document ingestion, retrieval relevance, authoritative-source ranking, role filtering, stale-source notices, conflicting policies, unsupported questions, citation validation, and redaction before a mocked provider request. The sample knowledge base includes product help and policy examples, as well as stale and resolved-ticket records to exercise conflict and authority handling. Live model groundedness is not asserted without provider credentials; the test validates citation filtering against a mocked model response instead.

## Known limitations

- This is a local demo, not a production multi-tenant service. The in-memory login sessions reset at restart, credentials are configured as environment JSON, and deployments need proper identity management, TLS, CSRF protections, rate limiting, and durable storage.
- Semantic retrieval requires an embeddings provider; otherwise the system uses BM25. The current vector store is in-memory and has no persistent embeddings index.
- Supported file types are JSON, Markdown, and text. PDFs, office files, OCR, streaming model responses, and production-grade policy conflict detection are not implemented.
- Common identifiers are redacted before configured model requests, but this is heuristic and cannot guarantee removal of every sensitive value. Apply your organization's approved data controls.
- Confidence and conflict cues are guardrails, not a substitute for agent review. Always confirm policy applicability and source freshness before responding.
