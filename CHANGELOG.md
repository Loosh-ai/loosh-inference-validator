# Changelog

All notable changes to loosh-inference-validator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
