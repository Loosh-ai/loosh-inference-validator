# Changelog

All notable changes to loosh-inference-validator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-02-12

### Fixed

- **Fiber MLTS inline re-negotiation with miners** — Validator now handles `401` and `400` responses from miners as key-expired signals, clears stale symmetric key cache, and re-handshakes immediately. Supports inline re-negotiation when the miner provides its RSA public key in the response body.
- **Fair miner selection across challenges** — Miner sampling is now per-challenge instead of per-availability-window. Previously, the availability worker pre-sampled `MAX_MINERS` from the available pool, and those same miners served all challenges for the next 30-second window. Now the full pool of available miners is returned and `main.py` randomly selects `MAX_MINERS` per challenge, giving every available miner a fair chance.
- **Removed misleading `MIN_STAKE_THRESHOLD`** — The constant was defined but never enforced in miner selection. Removed to avoid confusion. There is no minimum stake for miners.

## [1.2.0] - 2026-02-11

### Added

#### Hotkey Signature Authentication for Challenge API Calls

All outbound requests to the Challenge API are now signed with the validator's sr25519 hotkey (`X-Hotkey`, `X-Nonce`, `X-Signature` headers). This eliminates the need for a shared `CHALLENGE_API_KEY`. The signing message is `{nonce}:{hotkey}:{sha256(body)}` for requests with a body, or `{nonce}:{hotkey}` for body-less GET requests. Fully backwards-compatible — when the keypair is not yet loaded (startup race) or the Challenge API is an older version, falls back to the legacy `X-API-Key` header. Both headers are sent simultaneously during the transition period (`validator/network/challenge_api_auth.py`).

`CHALLENGE_API_KEY` is now **optional**. The startup validation no longer hard-fails when it is not set. An informational log is emitted instead, indicating that hotkey signature auth will be used. Validators upgrading to this version do not need to change their `.env` — existing API keys continue to work (`main.py`, `env.example`).

**`merge_auth_headers` utility** — Single call-site-friendly helper that picks the best available auth method (hotkey signature > API key) and enriches the existing headers dict. Sends both hotkey headers and `X-API-Key` simultaneously for maximum backwards compatibility with older Challenge API versions (`validator/network/challenge_api_auth.py`).

**Call sites updated** — All Challenge API call sites now use hotkey signature auth with API-key fallback:
- `validator_list_fetcher.py` — `GET /validators`
- `evaluation/sybil_sync.py` — `POST /analytics/sybil-detection/bulk`
- `evaluation/miner_network_reporter.py` — `POST /analytics/miner-network/bulk`
- `evaluation/set_weights.py` — `POST /analytics/sybil-scores/batch`, `POST /analytics/penalty-report`
- `evaluation/evaluation.py` — `POST /heatmap/upload` (multipart; signed without body hash)
- `challenge_api/update_challenge_response.py` — `POST /response/batch` (plain HTTP fallback)
- `scripts/generate_challenges.py` — `POST /challenge`

#### Automatic Validator Discovery

Validators are now **automatically discovered** by the Challenge API. Once a validator registers on subnet 78 and posts its IP and port to the chain via `fiber-post-ip`, the Challenge API detects the validator from the metagraph and begins sending challenges. Manual onboarding coordination is no longer required.

#### Centralized Internal Configuration (`validator/internal_config.py`)

**Rationale:** Ensure all validators use identical operational parameters for network consistency. Prevents accidental misconfiguration of critical values.

**New file:** `validator/internal_config.py` containing `InternalConfig` frozen dataclass with all hard-coded parameters:

| Category | Key Parameters |
|----------|---------------|
| **Miner Selection** | `MIN_MINERS=3`, `MAX_MINERS=10`, `MIN_STAKE_THRESHOLD=100`, `MAX_MINER_STAKE=999` |
| **Challenge Timing** | `CHALLENGE_INTERVAL_SECONDS=300`, `CHALLENGE_TIMEOUT_SECONDS=120`, `EVALUATION_TIMEOUT_SECONDS=300` |
| **Scoring** | `SCORE_THRESHOLD=0.7` |
| **Weight Setting** | `WEIGHTS_INTERVAL_SECONDS=4320`, `WEIGHT_MIN_SERVING_NODES=1`, `WEIGHT_FRESHNESS_HOURS=3/24` |
| **Deregistration** | `DEREGISTRATION_BLOCK_LIMIT=5000`, `DEGRADED_MODE_THRESHOLD=4000`, `EMERGENCY_MODE_THRESHOLD=4500` |
| **LLM Behavior** | `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_REQUEST_TIMEOUT_SECONDS` |
| **Embedding Model & Performance** | `SENTENCE_TRANSFORMER_MODEL=all-mpnet-base-v2`, `FALLBACK_MODEL=all-MiniLM-L6-v2`, `EMBEDDING_FP16_ENABLED`, `EMBEDDING_MAX_SEQ_LENGTH_DOC/SENTENCE`, `EMBEDDING_BATCH_SIZE_DOC/SENTENCE` |
| **Evaluation Quality** | Sentence-level relevance, coherence chain, prompt coverage, reasoning complexity weights |
| **Concurrency** | `MAX_CONCURRENT_CHALLENGES`, `MAX_CONCURRENT_AVAILABILITY_CHECKS`, `CHALLENGE_SEMAPHORE_TIMEOUT_SECONDS` |
| **Fiber MLTS** | `FIBER_KEY_TTL_SECONDS`, `FIBER_HANDSHAKE_TIMEOUT_SECONDS`, `FIBER_ENABLE_KEY_ROTATION` |
| **Sybil Detection** | Adaptive thresholds, multi-view fusion, trajectory analysis, group detection params |
| **Sybil Penalty** | `SYBIL_PENALTY_ENABLED`, `SYBIL_PENALTY_MAX`, `SYBIL_PENALTY_MIN_RETENTION`, `SYBIL_PENALTY_THRESHOLD`, `SYBIL_SAFETY_MAX_PENALIZED_FRACTION` |

**Removed from `.env`:** All parameters listed above are no longer configurable via environment variables.

**Migration:** No action required. Values match previous defaults. To modify, edit `validator/internal_config.py` directly.

#### Tiered Fallback Strategy for Weight Setting (`validator/evaluation/set_weights.py`)

**Problem:** Validators faced a critical dilemma — Challenge API failures → freshness gate zeros weights → weight setting skipped → validator deregistered at ~5000 blocks.

**Solution:** Three-tier operation mode based on blocks since last weight update:

| Mode | Blocks | Freshness Gate | Behavior |
|------|--------|---------------|----------|
| **NORMAL** | < 4000 | 3 hours | Standard quality, skip if all zero |
| **DEGRADED** | 4000–4499 | 24 hours | Relaxed gate, uses local DB emissions |
| **EMERGENCY** | ≥ 4500 | None | Uses all available emissions; uniform weights as absolute last resort |

Local emissions (stored during evaluation, regardless of Challenge API success) are used in degraded/emergency modes.

See [WEIGHT_SETTING_FALLBACK_STRATEGY.md](docs/WEIGHT_SETTING_FALLBACK_STRATEGY.md) for complete documentation.

#### Bittensor SDK Weight Setting (CRv4) (`validator/evaluation/set_weights.py`)

**Problem:** Fiber uses the deprecated `bittensor-commit-reveal` package (CRv3), which has been removed from the chain.

**Solution:** Weight setting now uses the Bittensor SDK v10+ (`subtensor.set_weights()`), which handles CRv4 automatically via `commit_timelocked_weights_extrinsic`. Fiber is still used for other chain operations where it works fine.

**New dependency:** `bittensor>=10.0.0` (added to `pyproject.toml`).

#### Enhanced Quality Scorer (`validator/evaluation/Evaluation/quality_scorer.py`)

**New file.** Provides embedding-aware, multi-granularity quality assessment replacing simple heuristic metrics:

1. **Multi-granularity relevance** — sentence-level + full-response cosine similarity to the prompt, with keyword coverage fallback
2. **Embedding-chain coherence** — local (adjacent), global (centroid-based), break counting, topic drift, and graph connectivity
3. **Prompt coverage completeness** — fraction of prompt semantic components addressed by the response
4. **Reasoning complexity** — semantic step clustering, shaped reward, non-redundancy, path length, curvature (gated by quality)
5. **Answer shape checks** — structural constraint detection (lists, code blocks, etc.)
6. **Keyword coverage** — TF-IDF-based keyword extraction fallback

Uses batch sentence embedding via the shared `SentenceTransformer` model for efficiency.

#### Garbage Consensus Prevention (`validator/evaluation/Evaluation/consensus_engine.py`)

Multi-layer defense against coordinated low-quality response attacks:

- **Semantic quality assessment**: Filters responses by prompt relevance, information density, specificity, and coherence (threshold: 0.35)
- **Smart outlier detection**: Quality-aware logic that protects high-quality unique responses from removal by garbage clusters
- **Quality-weighted consensus**: High-quality responses have more influence on consensus determination
- **Diversity bonus**: Rewards unique high-quality responses (up to +15%)
- **Garbage detection alerts**: Automatic logging when low-quality clusters detected

Pipeline now runs quality assessment **before** clustering (prevents garbage from forming consensus). Individual scoring rebalanced: similarity (40%), quality (25%), confidence (15%), consensus (10%), diversity (15%).

#### Advanced Sybil Detection (`validator/evaluation/sybil_detection.py`)

Major enhancements over the original implementation:


#### Sybil Penalty in Weight Setting (`validator/evaluation/set_weights.py`)

Graduated sybil penalty applied during weight setting:

#### Sybil Sync Background Task (`validator/evaluation/sybil_sync.py`)

Periodically syncs local sybil detection records to the Challenge API in batches:

#### Miner Network Reporter (`validator/evaluation/miner_network_reporter.py`)

**New file.** Background task that periodically reports miner network observations (IP, port, coldkey) to the Challenge API:

#### F3 Batch Response Submission (`validator/challenge_api/update_challenge_response.py`)

**New method:** `submit_response_batch()` replaces the deprecated single-response `update_challenge_response()`:

- Submits **all** miner responses and evaluation data in a single batch
- Prioritizes Fiber MLTS encryption with automatic HTTP fallback
- Includes consensus scores, emissions, sybil detection results, and narrative

**New models** (`validator/challenge_api/models.py`): `ResponseBatchSubmit`, `MinerResponseData`, `EvaluationResult`, `TokenUsage`.

#### Fiber MLTS Keypair Integration (`validator/network/fiber_client.py`)

- Module-level `set_validator_keypair()` / `get_validator_keypair()` so the sr25519 keypair is stored once at startup and shared by all `ValidatorFiberClient` instances
- `main.py` calls `set_validator_keypair(hotkey)` immediately after loading the wallet
- Handshake messages signed with `keypair.sign()` (sr25519) instead of placeholder signatures
- Removed `_load_private_key_from_hotkey()` placeholder and `private_key=None` constructor parameter

#### Sybil Detection Database Schema (`validator/db/schema.py`, `validator/db/operations.py`)

- **New table:** `SybilDetectionResult` with columns for suspicious pairs/groups (JSON), analysis report, thresholds, and challenge FK
- **New operations:** `log_sybil_detection_result()`, `delete_sybil_detection_result()`
- `Miner.hotkey` is now the primary persistent identifier; `node_id` (UID) is informational only
- `get_miner_ema_scores()` handles both old UID-keyed and new hotkey-keyed emissions for backward compatibility
- **UID compression safety**: `uid_to_hotkey` mapping now prefers the most recently updated miner when two share the same `node_id`

#### Embedding Model Upgrade to `sentence-transformers/all-mpnet-base-v2`

**Previous model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
**New model:** `sentence-transformers/all-mpnet-base-v2` (768-dim embeddings)

This is a significant quality improvement for consensus evaluation and sybil detection. MPNet produces substantially richer semantic representations — enabling more accurate similarity scoring, better outlier detection, and more reliable sybil pair identification. The tradeoff is ~3× slower encoding, but on GPU this remains well under 100ms per batch.

- Model name is now driven by `SENTENCE_TRANSFORMER_MODEL` in `internal_config.py` (no longer configurable via `.env`)
- `all-MiniLM-L6-v2` is retained as an automatic fallback if MPNet fails to load (e.g. disk space, download failure)
- FP16 optimization enabled by default for GPU acceleration (`EMBEDDING_FP16_ENABLED = True`)
- Configurable max sequence lengths for document (`EMBEDDING_MAX_SEQ_LENGTH_DOC`) and sentence (`EMBEDDING_MAX_SEQ_LENGTH_SENTENCE`) embeddings
- Configurable batch sizes for document and sentence embedding passes

#### IPv6 Address Support (`validator/miner_api/ipv6_fix.py`)

**New file.** `construct_server_address_with_ipv6()` utility:

- Detects IPv6 addresses (contains `:`) and wraps in square brackets: `http://[ipv6]:port`
- Maintains backward compatibility with IPv4
- Handles local development cases (`0.0.0.1`, `localhost`)

#### Heatmap & Quality Plot Optimization (`validator/evaluation/Evaluation/consensus_engine.py`)

- PIL/Pillow post-processing for lossless PNG compression (smaller file sizes)
- Adjusted figure size and font sizes for efficient images
- Quality plots gated behind `enable_semantic_quality` (no longer requires deprecated `apply_quality_filter`)

#### Validator Identity in Health Response (`validator/endpoints/availability.py`)

The `/health` and `/healthz` endpoints now return `service_name` and `version` in the `HealthResponse`:

- `service_name: "loosh-inference-validator"` — hard-coded identifier confirming this is Loosh validator software
- `version` — read from `importlib.metadata` (i.e. `pyproject.toml` version) at module load

The Challenge API uses these fields to verify that discovered validators are running the correct software at the required version before routing challenges. Validators that fail either check are marked `"outdated"` — excluded from challenge routing but continuously re-checked so they auto-recover on upgrade.

#### Firewall Whitelist Documentation (`README.md`, `docs/VALIDATOR_QUICKSTART.md`)

Added prominent **🔥 FIREWALL CONFIGURATION — REQUIRED** notice to both documents:

- Validators must whitelist inbound connections from `challenge.loosh.ai` (mainnet) and `challenge-test.loosh.ai` (testnet) on the port posted to the chain
- Includes UFW and iptables examples

### Changed

#### Configuration Split (`.env` → `internal_config.py`)

All operational parameters (scoring, timing, weights, evaluation, sybil, embedding) moved from `.env`-based `ValidatorConfig` to hard-coded `InternalConfig`. Only deployment-specific values remain in `.env`:

- API URLs, API keys, wallet paths, ports, device settings
- See `env.example` for the updated list

#### Consensus Engine Pipeline Reorder

Evaluation pipeline in `ConsensusEngine.evaluate_consensus()` reordered for garbage prevention:

1. Semantic quality assessment (new — first)
2. Semantic quality filter (new)
3. Smart outlier detection (new — replaces `_apply_outlier_filter`)
4. Clustering
5. Quality-weighted scoring (new)
6. Individual scoring (rebalanced weights)

#### `_apply_mask()` Now Maintains Quality State

`ConsensusEngine._apply_mask()` now also filters `quality_scores` and `quality_breakdowns` arrays alongside embeddings, confidences, responses, and labels. Previously these could become stale/misaligned after masking.

#### Weight Setting Uses Bittensor SDK

Replaced Fiber-based weight setting with Bittensor SDK `subtensor.set_weights()` for CRv4 compatibility. Substrate connections are now properly closed in `finally` blocks.

#### `challenge_timeout` Uses `INTERNAL_CONFIG`

`main.py` now reads `INTERNAL_CONFIG.CHALLENGE_TIMEOUT_SECONDS` instead of `getattr(config, 'challenge_timeout_seconds', 120)`.

### Fixed

#### `NameError: name 'filtered_count' is not defined` (`main.py`)

A variable was renamed from `filtered_count` to `total_excluded` during a refactor but one usage site in the challenge processing loop was missed, causing a crash on every challenge. Fixed by updating the stale reference and its log message. Additionally, `total_excluded` is now recalculated after the "no available nodes" retry loop so the debug log reflects the correct counts.

#### Granular Node-Filtering Logs (`main.py`)

The "No available nodes" and filtered-node debug messages now separately report self-excluded count, validator-db-excluded count, and the hotkeys of validator-db-excluded nodes. Previously these were conflated into a single "excluding N validator(s)" message, making it difficult to diagnose why a miner was being filtered out.

#### `ValidatorListFetcher` Respects `validator_permit` and `admin_approved` (`validator_list_fetcher.py`)

The fetcher now only adds a node to the internal `_validator_hotkeys` set if `validator_permit` is `true` **or** `metadata.admin_approved` is `true`. Previously it treated every entry in the Challenge API's validators table as a validator, which included miners that had been incorrectly auto-registered. Updated log message shows "permit or admin-approved" count vs total DB count.

#### Fiber sr25519 Signing (was placeholder)

**Problem:** `ValidatorFiberClient._perform_handshake()` noty always correctly authenticating.

**Fix:** Keypair loaded once at startup via `set_validator_keypair()`, all handshake messages now signed with `keypair.sign()` (sr25519).

#### Pipeline Timing Import Fix (`validator/challenge/send_challenge.py`)

**Problem:** `name 'PipelineTiming' is not defined` runtime error.

**Fix:** Added missing imports: `from validator.timing import PipelineTiming, PipelineStages`

#### Fiber Key Expiration Race Condition (`validator/network/fiber_client.py`)

**Problem:** 401 errors on heatmap uploads due to race condition between client and server key expiration.

**Fix:** Added 60-second safety margin for client-side key refresh. Added automatic retry with re-handshake on 401 errors. Improved key age logging.

#### Test Mode Response Detection (`validator/evaluation/evaluation.py`)

**Problem:** Miners could return test mode responses and still receive emissions.

**Fix:** Test mode responses are filtered out before evaluation.

#### `blocks_since_update` Safety in DEGRADED Mode (`validator/evaluation/set_weights.py`)

**Problem:** `blocks_since_update` could be `None` when metagraph fetch fails, causing `TypeError` in DEGRADED mode arithmetic.

**Fix:** Added `None` guard with graceful fallback messaging.

#### Atomic Sybil Cache File Write (`validator/evaluation/set_weights.py`)

**Problem:** `_save_file_cache()` wrote directly to the cache file — a crash mid-write could corrupt it.

**Fix:** Writes to a temporary file first, then uses `os.replace()` for atomic rename.

#### UID-to-Hotkey Collision in EMA Scores (`validator/db/operations.py`)

**Problem:** When two miners share the same `node_id` (UID compression), the `uid_to_hotkey` mapping could attribute old emissions to the wrong hotkey.

**Fix:** Mapping now tracks `last_updated` timestamp and prefers the most recently updated miner for each UID.

#### scikit-learn `AgglomerativeClustering` Compatibility (`validator/evaluation/Evaluation/consensus_engine.py`)

**Problem:** `TypeError` with `affinity` parameter in scikit-learn >= 1.2.

**Fix:** Try/except to use `metric='precomputed'` (>= 1.2) with fallback to `affinity='precomputed'`.

---

## [1.0.1] - 2026-02-05

### Fixed

#### SUBTENSOR_ADDRESS Configuration Fix (`validator/main.py`)

**Problem:** Calls to substrate not always using SUBTENSOR_ADDRESS

This meant validators with custom `SUBTENSOR_ADDRESS` configurations would silently fall back to Fiber's default endpoint for that network. If the default endpoint was unreachable or had DNS issues, the validator would fail with `[Errno -2] Name or service not known`, even when a working custom endpoint was configured.

**Solution:** Both calls now properly pass the configured `subtensor_address`:

#### IP Filtering Edge Case (`validator/evaluation/set_weights.py`)

**Problem:** The IP check `if not node.ip` didn't handle nodes with IP stored as `"0"` (string zero from chain integer 0).

**Solution:** Enhanced check to filter out `"0"` and `"0.0.0.0"` as invalid IPs:

### Changed

#### Weight Setting Interval Now Hard-Coded (`validator/evaluation/set_weights.py`, `validator/config.py`)

**Rationale:** Ensure all validators set weights on the same schedule for network consistency, ensure changes can be made to that duration by git pull. Value set to 72 minutes.

**Changes:**
- Removed `WEIGHTS_INTERVAL_SECONDS` environment variable
- Removed `weights_interval_seconds` field from `ValidatorConfig`
- Removed `weights_interval` property method from `ValidatorConfig`
- Added hard-coded constant `WEIGHTS_INTERVAL_SECONDS = 4320` (72 minutes) in `set_weights.py`

---

### Added

#### Serving Miner Filtering (`validator/evaluation/set_weights.py`)

Weight setting now filters nodes to only include **serving miners**, excluding:

| Filter | Reason |
|--------|--------|
| Validator's own node | Self-exclusion |
| Stake >= 999 TAO | High stake indicates validator, not miner |
| No IP/port advertised | Node not serving |
| In validator database | Registered validators from Challenge API |

**New function:** `_filter_serving_miners()` with detailed statistics logging.

#### Freshness Gate for Weight Setting (`validator/evaluation/set_weights.py`, `validator/db/operations.py`)

Miners without recent successful responses now receive **zero weight**, preventing stale miners from accumulating weight based on old performance.

**Internal constants (in `set_weights.py`):**
- `WEIGHT_FRESHNESS_HOURS = 3` - Hours before a miner is considered stale
- `WEIGHT_MIN_SERVING_NODES = 1` - Minimum serving miners required to set weights

**New database method:** `get_miner_last_success_times()` - Retrieves last successful response timestamps for all miners.

#### Skip Behavior for Zero Weights (`validator/evaluation/set_weights.py`)

**CRITICAL CHANGE:** When all EMA scores are zero (after freshness gate), weight setting is now **skipped entirely** instead of distributing weights uniformly.

This prevents rewarding untested, down, or adversarial miners during:
- Validator startup (no evaluations yet)
- Database outages
- All miners being stale

#### Validator List Integration (`validator/evaluation/set_weights.py`, `validator/main.py`)

The `set_weights()` function now accepts an optional `validator_list_fetcher` parameter to exclude nodes registered as validators in the Challenge API database.

#### Enhanced Logging (`validator/evaluation/set_weights.py`)

All log messages now include `[set_weights]` prefix for easy filtering. New statistics logged:
- Node filtering breakdown (total, self, high stake, no endpoint, validator DB, included)
- Freshness gate results (stale miners zeroed)
- Weight distribution (min/max/avg, miners with weight)

---

### Technical Improvements

#### Async Event Loop Fix (`validator/evaluation/set_weights.py`)

**Changed:** `asyncio.get_event_loop()` → `asyncio.get_running_loop()`

This is safer in Python 3.10+ and properly detects when called outside a running event loop.

---

## Migration Notes

### Environment Variables

**Removed:**
- `WEIGHTS_INTERVAL_SECONDS` - No longer configurable (hard-coded to 4320s)


### Existing Validators

No action required. Changes are backward compatible:
- Existing on-chain weights are preserved during zero-score periods
- New filtering is additive (more restrictive, not less)
- New config fields have sensible defaults

---

## [1.0.0] - 2026-02-01

Initial production release of loosh-inference-validator.

### Features
- EMA-based miner scoring with configurable lookback and alpha
- Fiber-based miner querying with MLTS encryption
- Challenge API integration for receiving challenges
- SQLAlchemy-based persistence for miner statistics
- CRv4-compatible weight setting via Bittensor SDK v10+
- Custom subtensor endpoint support (`SUBTENSOR_ADDRESS`)
- Validator list caching from Challenge API

---

[1.2.0]: https://github.com/Loosh-ai/loosh-inference-validator/compare/v1.0.1...v1.2.0
[1.0.1]: https://github.com/Loosh-ai/loosh-inference-validator/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Loosh-ai/loosh-inference-validator/releases/tag/v1.0.0
