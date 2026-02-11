# Weight Setting Fallback Strategy

## Overview

The Loosh Inference Validator implements a **tiered fallback strategy** for weight setting to prevent validator deregistration while maintaining fair reward distribution. This system addresses the challenge of Challenge API outages without compromising the integrity of miner rewards.

## The Problem

Validators face a critical tension:

1. **Deregistration Risk**: Bittensor deregisters validators who don't set weights for ~5000 blocks
2. **Fair Rewards**: Weights should reflect actual miner performance, not arbitrary distribution
3. **Challenge API Dependency**: Validators evaluate miners locally but may fail to submit results to Challenge API

**Key Insight**: Validators **already store emissions locally** after evaluation, regardless of whether the Challenge API accepts the submission. The weight-setting system uses these **local emissions**, not Challenge API confirmation.

## Solution: Tiered Fallback Modes

The validator operates in three modes based on `blocks_since_last_update`:

### 1. NORMAL MODE (< 4000 blocks)

**When**: Validator is current on weight updates (< 80% of deregistration threshold)

**Behavior**:
- Calculate EMA scores from local DB emissions (24-hour lookback, α=0.3)
- Apply **3-hour freshness gate**: miners without recent successful responses get zero weight
- If all weights are zero → **SKIP** weight setting (preserve on-chain weights)
- This is the standard, production-grade behavior

**Rationale**: In normal operation, we want strict quality standards. Skipping weight setting when all miners are stale is safe because the validator has plenty of time before deregistration.

```python
WEIGHT_FRESHNESS_HOURS = 3
blocks_since_update < 4000 → NORMAL MODE
```

### 2. DEGRADED MODE (4000-4499 blocks)

**When**: Approaching deregistration (80-90% of threshold)

**Behavior**:
- Calculate EMA scores from local DB emissions
- Apply **24-hour relaxed freshness gate** (instead of 3 hours)
- Use emissions from local DB even if Challenge API was unavailable
- **STILL SKIP** if all weights are zero after relaxed gate
- Log warnings to alert operators

**Rationale**: Challenge API issues shouldn't cause deregistration. Miners who responded successfully within 24 hours (even if Challenge API rejected the batch) should still receive credit based on local evaluation data.

```python
WEIGHT_FRESHNESS_HOURS_DEGRADED = 24
4000 ≤ blocks_since_update < 4500 → DEGRADED MODE
```

**Example Log**:
```
[set_weights] ⚠️ DEGRADED MODE: 4123 blocks since last update (877 blocks until deregistration at 5000).
Will use relaxed freshness gate (24h instead of 3h).
```

### 3. EMERGENCY MODE (≥ 4500 blocks)

**When**: Critical deregistration risk (90%+ of threshold)

**Behavior**:
- Calculate EMA scores from local DB emissions
- **NO freshness gate**: use ALL available emissions regardless of age
- If still no emissions → distribute **minimal uniform weights** to all serving miners
- This is absolute last resort to prevent deregistration
- Log critical alerts for operator intervention

**Rationale**: Deregistration costs validators their registration and disrupts the network. Emergency mode prioritizes survival over perfect fairness, using historical data or uniform distribution as a last resort.

```python
blocks_since_update ≥ 4500 → EMERGENCY MODE
```

**Example Logs**:
```
[set_weights] ⚠️ EMERGENCY MODE: 4523 blocks since last update (477 blocks until deregistration at 5000).
Will use ANY available emissions to prevent deregistration!

[set_weights] EMERGENCY MODE: All EMA scores are zero but must set weights to prevent deregistration.
Distributing uniform minimal weights to all 42 serving miners as absolute last resort.
```

## Constants

All thresholds are hard-coded in `validator/evaluation/set_weights.py` to ensure consistent behavior across all validators:

```python
# Freshness gates
WEIGHT_FRESHNESS_HOURS = 3                    # Normal mode
WEIGHT_FRESHNESS_HOURS_DEGRADED = 24          # Degraded mode

# Deregistration safety thresholds
DEREGISTRATION_BLOCK_LIMIT = 5000             # Bittensor's deregistration threshold
DEGRADED_MODE_THRESHOLD = 4000                # 80% of limit
EMERGENCY_MODE_THRESHOLD = 4500               # 90% of limit

# Other constants
WEIGHTS_INTERVAL_SECONDS = 4320               # 72 minutes
EMA_LOOKBACK_HOURS = 24                       # EMA calculation window
EMA_ALPHA = 0.3                               # EMA smoothing factor
```

## Emissions Recording

**Critical Detail**: Emissions are recorded in the validator's local database immediately after evaluation, **before** attempting to submit to the Challenge API.

**Flow**:
```
1. Validator evaluates miner responses → calculates emissions
2. Validator.log_evaluation_result() → saves emissions to local DB
3. Validator submits batch to Challenge API → may succeed or fail
4. Weight setting uses local DB emissions (from step 2)
```

**Location**: `validator/evaluation/evaluation.py` lines 378-384
```python
# Log evaluation result TO LOCAL DB
self.db_manager.log_evaluation_result(
    challenge_id=challenge_id,
    consensus_score=consensus_score_float,
    heatmap_path=heatmap_path,
    narrative=narrative,
    emissions=emissions_float
)
```

This means:
- ✅ Emissions are **always available** for weight calculation (unless DB fails)
- ✅ Challenge API outages **don't lose emission data**
- ✅ Freshness gate determines whether to **use** the emissions, not whether they exist

## Handling Challenge API Failures

### Scenario 1: Challenge API Down for < 3 Hours

**Mode**: NORMAL → weights continue normally
**Behavior**: Local emissions are fresh, weights calculated from local data
**Impact**: None - this is transparent to weight setting

### Scenario 2: Challenge API Down for 3-24 Hours

**Mode**: NORMAL → DEGRADED (if blocks accumulate)
**Behavior**:
- NORMAL: Freshness gate zeros out stale miners, weight setting skipped
- Eventually: Blocks accumulate, enters DEGRADED mode
- DEGRADED: Relaxed 24-hour gate allows using recent local emissions
- Weights resume based on actual performance (from local DB)

**Impact**: Temporary zero-weight period, then recovery

### Scenario 3: Challenge API Down for > 24 Hours + Critical Block Count

**Mode**: DEGRADED → EMERGENCY
**Behavior**:
- DEGRADED: Still looking for 24-hour-fresh emissions
- EMERGENCY: Use ANY emissions, or uniform distribution if necessary
- Prevents deregistration

**Impact**: Weights may not reflect most recent performance, but validator survives

### Scenario 4: Complete System Failure (No Local Emissions)

**Mode**: EMERGENCY (if blocks > 4500)
**Behavior**: Distribute uniform minimal weights to prevent deregistration
**Impact**: Short-term unfair distribution, but:
- Validator stays registered
- Can recover when system restarts
- Better than losing validator slot entirely

## Monitoring & Alerts

Validators should monitor for these log patterns:

**Warning Signs**:
```
[set_weights] ⚠️ DEGRADED MODE: 4123 blocks since last update
```
→ Action: Check Challenge API connectivity, investigate why submissions are failing

**Critical Alerts**:
```
[set_weights] ⚠️ EMERGENCY MODE: 4678 blocks since last update
[set_weights] EMERGENCY MODE: All EMA scores are zero but must set weights to prevent deregistration
```
→ Action: URGENT - Validator is about to be deregistered, investigate immediately

**Recovery**:
```
[set_weights] NORMAL MODE: 45 blocks since last update (4955 blocks until deregistration threshold).
[set_weights] [NORMAL MODE] SUCCESS: Set weights on chain for 38 miners
```
→ Validator recovered and returned to normal operation

## Why This Approach Doesn't Skew Weights

1. **Local Emissions Are Real**: Validators evaluate miners locally using the same consensus algorithm regardless of Challenge API
2. **Challenge API is for Aggregation**: The Challenge API aggregates results across validators but doesn't determine individual validator's weight decisions
3. **Freshness Gates Prevent Stale Data**: Even in degraded mode, we enforce recency requirements (24h)
4. **Emergency Mode is Rare**: Uniform distribution only happens as absolute last resort when:
   - No emissions exist in DB (system failure), AND
   - Deregistration is imminent (4500+ blocks)
5. **Mode Transitions are Gradual**: Validators don't jump to emergency mode; they degrade through normal → degraded → emergency

## Comparison to Previous Behavior

### Before (Issue State):
```
Challenge API down for > 3 hours
→ All miners marked stale
→ All weights zero
→ Weight setting SKIPPED indefinitely
→ Validator deregistered at 5000 blocks
```

### After (With Fallback Strategy):
```
Challenge API down for > 3 hours
→ Validator enters DEGRADED mode at 4000 blocks
→ Uses relaxed 24-hour freshness gate
→ Sets weights based on local emissions (real performance data)
→ Validator continues operating
→ Returns to NORMAL mode when Challenge API recovers
```

## Edge Cases

### Q: What if local DB fails?
**A**: Validator has bigger problems. Emergency mode will distribute uniform weights if deregistration is imminent, but primary fix is restoring DB.

### Q: Will miners game the system knowing about degraded mode?
**A**: No, because:
1. Emissions are based on actual response quality/consensus
2. Degraded mode still uses real evaluation data (not fake rewards)
3. Mode thresholds are based on validator state, not miner behavior

### Q: Can validators manipulate weights by staying in EMERGENCY mode?
**A**: No, because:
1. Emergency mode requires 4500+ blocks since last update
2. After setting weights, counter resets, returns to NORMAL
3. Can't stay in emergency mode continuously
4. Uniform distribution doesn't benefit any specific miner

### Q: What if a miner hasn't responded in weeks but validator is in EMERGENCY mode?
**A**: They still receive zero weight in NORMAL/DEGRADED modes (covered by EMA lookback window of 24 hours). In EMERGENCY mode, if they're a serving miner, they may get minimal uniform weight temporarily - but this is preferable to validator deregistration.

## Testing Recommendations

1. **Monitor mode transitions** in production logs
2. **Alert on DEGRADED mode** entry (indicates Challenge API issues)
3. **Page on EMERGENCY mode** entry (critical intervention needed)
4. **Track blocks_since_last_update** metric in observability dashboards
5. **Verify weight distributions** remain consistent before/after Challenge API outages

## Configuration

All configuration is hard-coded in `validator/evaluation/set_weights.py` to ensure network-wide consistency. **Do not configure these values via environment variables** - changes must be deployed via git pull to all validators simultaneously.

To modify thresholds, edit the constants at the top of `set_weights.py` and deploy to all validators.
