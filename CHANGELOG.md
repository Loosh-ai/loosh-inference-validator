# Changelog

All notable changes to loosh-inference-validator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

#### Centralized Internal Configuration (`validator/internal_config.py`)

**Rationale:** Ensure all validators use identical operational parameters for network consistency. Prevents accidental misconfiguration of critical values.

**New file:** `validator/internal_config.py` containing `InternalConfig` dataclass with all hard-coded parameters:

| Category | Parameters |
|----------|------------|
| **Miner Selection** | `MIN_MINERS=3`, `MAX_MINERS=10`, `MIN_STAKE_THRESHOLD=100`, `MAX_MINER_STAKE=999` |
| **Challenge Timing** | `CHALLENGE_INTERVAL_SECONDS=300`, `CHALLENGE_TIMEOUT_SECONDS=120`, `EVALUATION_TIMEOUT_SECONDS=300` |
| **Scoring** | `SCORE_THRESHOLD=0.7` |
| **Weight Setting** | `WEIGHTS_INTERVAL_SECONDS=4320`, `WEIGHT_MIN_SERVING_NODES=1`, `WEIGHT_FRESHNESS_HOURS=3/24` |
| **Deregistration** | `DEREGISTRATION_BLOCK_LIMIT=5000`, `DEGRADED_MODE_THRESHOLD=4000`, `EMERGENCY_MODE_THRESHOLD=4500` |

**Removed from `.env`:** All parameters listed above are no longer configurable via environment variables.

**Migration:** No action required. Values match previous defaults. To modify, edit `validator/internal_config.py` directly.

### Added

#### Tiered Fallback Strategy for Weight Setting (`validator/evaluation/set_weights.py`, `docs/WEIGHT_SETTING_FALLBACK_STRATEGY.md`)

**Problem:** Validators faced a critical dilemma:
1. Challenge API failures → validators can't submit results → freshness gate zeros weights → weight setting skipped
2. Skipping weight setting indefinitely → validator deregistered at ~5000 blocks
3. BUT: Distributing weights uniformly would reward untested/adversarial miners unfairly

**Solution:** Implemented three-tier operation mode based on blocks since last weight update:

**NORMAL MODE** (< 4000 blocks):
- Standard 3-hour freshness gate
- Skip weight setting if all weights zero (safe because plenty of time before deregistration)
- Production-grade quality standards

**DEGRADED MODE** (4000-4499 blocks):
- Relaxed 24-hour freshness gate (instead of 3 hours)
- Uses emissions from **local validator DB** (stored during evaluation, regardless of Challenge API success)
- Still skips if all weights zero after relaxed gate
- Logs warnings to alert operators

**EMERGENCY MODE** (≥ 4500 blocks):
- No freshness gate - uses ALL available emissions from local DB
- If still no emissions → distributes minimal uniform weights as absolute last resort
- Critical alerts for operator intervention

**Key Insight:** Validators **already store emissions locally** after evaluation (before submitting to Challenge API). The freshness gate determines whether to **use** the emissions, not whether they exist.

**Why This Doesn't Skew Weights:**
- Local emissions are real (same consensus algorithm as Challenge API submission)
- Degraded/Emergency modes use actual performance data from local evaluations
- Uniform distribution only happens in extreme edge case (no local DB emissions + imminent deregistration)
- Mode transitions are gradual (normal → degraded → emergency)

**Constants** (hard-coded for network consistency):
```python
WEIGHT_FRESHNESS_HOURS = 3                    # Normal mode
WEIGHT_FRESHNESS_HOURS_DEGRADED = 24          # Degraded mode
DEREGISTRATION_BLOCK_LIMIT = 5000             # Bittensor threshold
DEGRADED_MODE_THRESHOLD = 4000                # 80% of limit
EMERGENCY_MODE_THRESHOLD = 4500               # 90% of limit
```

**Monitoring:**
- `⚠️ DEGRADED MODE` logs → Check Challenge API connectivity
- `⚠️ EMERGENCY MODE` logs → URGENT intervention needed
- Track `blocks_since_last_update` in observability dashboards

See [WEIGHT_SETTING_FALLBACK_STRATEGY.md](docs/WEIGHT_SETTING_FALLBACK_STRATEGY.md) for complete documentation.

#### Garbage Consensus Prevention (`validator/evaluation/Evaluation/consensus_engine.py`, `validator/evaluation/evaluation.py`)

Added multi-layer defense against coordinated low-quality response attacks:

- **Semantic quality assessment**: Filters responses by prompt relevance, information density, specificity, and coherence (threshold: 0.35)
- **Smart outlier detection**: Quality-aware logic that protects high-quality unique responses from removal
- **Quality-weighted consensus**: High-quality responses have more influence on consensus determination
- **Diversity bonus**: Rewards unique high-quality responses (up to +15%)
- **Garbage detection alerts**: Automatic logging when low-quality clusters detected

Pipeline now runs quality assessment FIRST before clustering (prevents garbage from forming consensus). Individual scoring rebalanced: similarity (40%), quality (25%), confidence (15%), consensus (10%), diversity (15%).

### Fixed

#### Pipeline Timing Import Fix (`validator/challenge/send_challenge.py`)

**Problem:** Runtime error when trying to merge miner timing data:
```
name 'PipelineTiming' is not defined
```

**Solution:** Added missing imports:
```python
from validator.timing import PipelineTiming, PipelineStages
```

**Impact:**
- ✅ Miner timing data can now be properly merged into pipeline timing
- ✅ Timing metadata from miner responses is correctly captured
- ✅ No more runtime errors when miners return timing information

#### Fiber Encryption Key Expiration Race Condition (`validator/network/fiber_client.py`)

**Problem:**  401 errors when uploading heatmaps to Challenge API:
```
Failed to decrypt payload. Key may be expired or invalid.
```

**Root Cause:** Race condition between client and server key expiration checks:

**Solution:** 
- Added 60-second **safety margin** for client-side key refresh (`key_refresh_margin_seconds = 60`)
- Client now refreshes keys at 3540s (TTL - margin) instead of 3600s
- Added automatic retry with re-handshake on 401 errors in `send_encrypted_upload`
- Improved logging to show key age and effective TTL when refreshing

**Impact:**
- ✅ Prevents keys from expiring during transmission
- ✅ Reduces fallback to plain HTTP uploads
- ✅ Automatic recovery from occasional race conditions
- ✅ Better debugging with detailed key age logging

#### Test Mode Response Detection (`validator/evaluation/evaluation.py`)

**Problem:** Miners could return test mode responses and still receive emissions/rewards:
**Solution:**
Test Mode Responses are filtered out.

#### IPv6 Address Support

**Problem:** Validators couldn't connect to miners with IPv6 addresses

**Solution:** Created `construct_server_address_with_ipv6()` utility function that:
- Detects IPv6 addresses (contains `:`) and wraps them in square brackets: `http://[ipv6]:port`
- Maintains backward compatibility with IPv4 addresses: `http://ipv4:port`
- Handles local development cases (`0.0.0.1`, `localhost`)

**Impact:**
- ✅ Validators can now connect to IPv6 miners for availability checks
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

[Unreleased]: https://github.com/Loosh-ai/loosh-inference-validator/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/Loosh-ai/loosh-inference-validator/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Loosh-ai/loosh-inference-validator/releases/tag/v1.0.0
