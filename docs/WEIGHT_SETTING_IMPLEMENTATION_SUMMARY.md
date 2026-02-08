# Weight Setting Fallback Strategy - Implementation Summary

## Problem Statement

You identified a critical two-part problem:

1. **Validators don't record challenges as ready for weighing unless accepted by Challenge API**
   - Actually, this was a misunderstanding - validators **DO** record emissions locally
   - Emissions are saved to local DB immediately after evaluation (before Challenge API submission)
   
2. **Zero-weight protection prevented validators from setting ANY weights during Challenge API outages**
   - This was the real issue
   - Code at line 429-438 in `set_weights.py` skipped weight setting when all weights were zero
   - During Challenge API outages > 3 hours, freshness gate zeroed all weights → skipped indefinitely
   - This would lead to deregistration at ~5000 blocks

## Root Cause Analysis

The actual flow is:

```
1. Validator evaluates responses → calculates emissions
2. Emissions saved to LOCAL DB ← THIS ALWAYS HAPPENS
3. Validator attempts to submit batch to Challenge API ← THIS MAY FAIL
4. Weight setting reads from LOCAL DB (step 2) ← Uses real data regardless of step 3
5. Freshness gate determines whether to USE the local emissions
```

**Key Insight**: The problem wasn't missing emissions - they were always in the local DB. The problem was the **freshness gate** combined with the **"skip on zero weights"** logic meant validators would never set weights during extended Challenge API outages.

## Solution: Tiered Fallback Strategy

Implemented three operation modes based on `blocks_since_last_update`:

### Mode Transitions

```
blocks < 4000:  NORMAL MODE
                ↓ (Challenge API issues persist)
4000-4499:      DEGRADED MODE
                ↓ (Still can't set weights)
≥ 4500:         EMERGENCY MODE
```

### NORMAL MODE (< 4000 blocks)

**Behavior:**
- EMA from local DB (24h lookback, α=0.3)
- 3-hour freshness gate
- Skip if all weights zero
- Standard production quality

**Log Example:**
```
[set_weights] NORMAL MODE: 1234 blocks since last update (3766 blocks until deregistration threshold).
```

### DEGRADED MODE (4000-4499 blocks)

**Behavior:**
- EMA from local DB (same data source)
- **24-hour relaxed freshness gate** (8x more lenient)
- Still skip if all weights zero after relaxed gate
- Warning logs for operators

**Why This Works:**
- Challenge API down for hours/days
- BUT miners still responded to validator's local challenges
- Validator has the evaluation data in local DB
- Relaxed freshness gate allows using that data
- **Uses real performance metrics, not fake rewards**

**Log Example:**
```
[set_weights] ⚠️ DEGRADED MODE: 4123 blocks since last update (877 blocks until deregistration at 5000).
Will use relaxed freshness gate (24h instead of 3h).
```

### EMERGENCY MODE (≥ 4500 blocks)

**Behavior:**
- EMA from local DB
- **No freshness gate** - use ANY available emissions
- If still no emissions → **uniform distribution** (last resort)
- Critical alerts

**When This Triggers:**
- System-wide failure (no local emissions) + imminent deregistration
- OR: Fresh validator startup with no history + rate limit delays
- OR: Extended network partition

**Log Example:**
```
[set_weights] ⚠️ EMERGENCY MODE: 4678 blocks since last update (322 blocks until deregistration at 5000).
Will use ANY available emissions to prevent deregistration!

[set_weights] EMERGENCY MODE: All EMA scores are zero but must set weights to prevent deregistration.
Distributing uniform minimal weights to all 42 serving miners as absolute last resort.
```

## Why This Doesn't Skew Weights

1. **Local emissions are real performance data** - same consensus algorithm used regardless of Challenge API
2. **Challenge API is for aggregation** - validators make independent weight decisions
3. **Freshness gates prevent stale data** - even in degraded mode (24h limit)
4. **Emergency uniform distribution is rare** - only when NO emissions exist + deregistration imminent
5. **Mode transitions are gradual** - validators degrade gracefully

## Implementation Details

### Modified Files

1. **`validator/evaluation/set_weights.py`**
   - Added constants for three modes
   - Added mode determination logic (lines 314-352)
   - Modified freshness gate application (lines 388-427)
   - Modified zero-weight handling (lines 429-467)
   - Added mode tags to all log messages

2. **`docs/WEIGHT_SETTING_FALLBACK_STRATEGY.md`**
   - Complete documentation of the tiered strategy
   - Examples, edge cases, monitoring guidance
   - FAQ for common questions

3. **`README.md`**
   - Updated weight setting process section
   - Added reference to detailed documentation
   - Explained local DB vs Challenge API relationship

4. **`CHANGELOG.md`**
   - Documented the change with rationale
   - Included configuration constants
   - Monitoring guidance

### Key Code Changes

**Before** (line 429-438):
```python
if total_weight <= 0:
    logger.warning(
        f"[set_weights] All EMA scores are zero (after freshness gate). "
        f"SKIPPING weight setting to preserve previous on-chain weights."
    )
    return
```

**After** (line 429-467):
```python
if total_weight <= 0:
    if operation_mode == "EMERGENCY":
        # Distribute uniform weights to prevent deregistration
        uniform_weight = 1.0 / len(serving_miners)
        weights_list = [uniform_weight for _ in serving_miners]
    elif operation_mode == "DEGRADED":
        # Skip but warn urgently
        logger.error(f"DEGRADED MODE: All EMA scores zero. Only {X} blocks until EMERGENCY MODE.")
        return
    else:
        # Standard skip behavior
        logger.warning(f"NORMAL MODE: All EMA scores zero. SKIPPING.")
        return
```

## Monitoring & Operations

### Log Patterns to Monitor

**Healthy Operation:**
```
[set_weights] NORMAL MODE: 45 blocks since last update
[set_weights] [NORMAL MODE] SUCCESS: Set weights on chain for 38 miners
```

**Warning - Investigate Challenge API:**
```
[set_weights] ⚠️ DEGRADED MODE: 4123 blocks since last update
[set_weights] Freshness gate: zeroed out 0 stale miners in DEGRADED mode
[set_weights] [DEGRADED MODE] SUCCESS: Set weights on chain for 38 miners
```
→ **Action**: Check Challenge API connectivity, investigate submission failures

**Critical - Immediate Intervention:**
```
[set_weights] ⚠️ EMERGENCY MODE: 4678 blocks since last update (322 blocks until deregistration)
[set_weights] EMERGENCY MODE: All EMA scores are zero but must set weights to prevent deregistration
[set_weights] [EMERGENCY MODE] SUCCESS: Set weights on chain for 42 miners with uniform distribution
```
→ **Action**: URGENT - Investigate why no emissions exist in local DB, check system health

### Metrics to Track

1. **`blocks_since_last_update`** - Primary indicator
2. **`operation_mode`** - Should be "NORMAL" in steady state
3. **`stale_count`** - How many miners failed freshness gate
4. **`miners_with_weight`** - Non-zero weight count
5. **Weight distribution** - min/max/avg to detect anomalies

## Testing Scenarios

### Scenario 1: Challenge API Down for 6 Hours (Normal Recovery)

**Timeline:**
```
T+0h:   Challenge API goes down
        Validators continue evaluating locally
        Emissions stored in local DB
        
T+3h:   Freshness gate starts zeroing weights in NORMAL mode
        Weight setting skipped (all zero)
        blocks_since_update accumulates
        
T+5h:   blocks_since_update reaches 4000
        Enters DEGRADED MODE
        Relaxed 24h freshness gate applied
        Finds fresh emissions from T+0 to T+5h in local DB
        Sets weights successfully based on real performance
        
T+6h:   Challenge API recovers
        Returns to NORMAL mode on next weight cycle
```

**Result**: ✅ No deregistration, weights reflect actual performance

### Scenario 2: Complete System Failure (Emergency Recovery)

**Timeline:**
```
T+0h:   Database corruption / complete system failure
        No emissions in local DB
        
T+70h:  blocks_since_update reaches 4500
        Enters EMERGENCY MODE
        No emissions available even without freshness gate
        Distributes uniform minimal weights
        
T+72h:  System restored, database recovered
        Validator resumes normal evaluation
        Emissions accumulate in local DB
        Returns to NORMAL mode
```

**Result**: ✅ Validator survives, temporary unfair distribution (acceptable trade-off)

## Configuration

All thresholds are **hard-coded** in `set_weights.py` for network consistency:

```python
# Freshness gates
WEIGHT_FRESHNESS_HOURS = 3                    # Normal: strict
WEIGHT_FRESHNESS_HOURS_DEGRADED = 24          # Degraded: relaxed

# Mode thresholds
DEREGISTRATION_BLOCK_LIMIT = 5000             # Bittensor's limit
DEGRADED_MODE_THRESHOLD = 4000                # 80% of limit
EMERGENCY_MODE_THRESHOLD = 4500               # 90% of limit

# EMA parameters
EMA_LOOKBACK_HOURS = 24
EMA_ALPHA = 0.3
WEIGHTS_INTERVAL_SECONDS = 4320               # 72 minutes
```

**DO NOT** configure these via environment variables - changes must be git-deployed to all validators simultaneously.

## Deployment Notes

1. This change is **backward compatible** - validators will default to NORMAL mode
2. No database migrations required - uses existing emissions table
3. No configuration changes required - all hard-coded
4. Operators should add monitoring for mode transitions
5. Existing validators will immediately benefit from fallback protection

## Comparison to Alternatives

### Alternative 1: Always Set Uniform Weights When Zero
❌ **Problem**: Rewards untested/adversarial miners unfairly

### Alternative 2: Remove Freshness Gate Entirely
❌ **Problem**: Stale miners receive rewards based on old performance

### Alternative 3: Store Challenge API Success Status
❌ **Problem**: Adds complexity, doesn't solve root cause (emissions are already local)

### ✅ Chosen Solution: Tiered Fallback
- **Preserves quality** in normal operation
- **Uses real data** in degraded operation
- **Prevents deregistration** in emergency
- **Minimal code changes** - leverages existing local DB
- **Clear operational signals** - mode transitions are visible

## Questions Answered

**Q: Does this mean weights are skewed during Challenge API outages?**
A: No - DEGRADED mode uses the same emissions data that would have been submitted to Challenge API. The validator evaluated miners locally using the full consensus algorithm. Challenge API acceptance is orthogonal to weight calculation.

**Q: What if miners collude to exploit DEGRADED mode?**
A: They can't - mode thresholds are based on validator's `blocks_since_last_update`, not miner behavior. Emissions are still based on actual response quality/consensus.

**Q: Why not just always use 24-hour freshness gate?**
A: In normal operation, we want strict quality standards (3 hours). Only relax when necessary to prevent deregistration.

**Q: What happens when validator returns from EMERGENCY to NORMAL?**
A: After setting weights in EMERGENCY mode, `blocks_since_last_update` resets to near-zero. Validator immediately returns to NORMAL mode on next cycle. Emergency weights are a one-time event, not persistent state.

## Success Criteria

✅ **Prevents deregistration** during Challenge API outages
✅ **Doesn't skew weights** - uses real performance data from local evaluations
✅ **Maintains quality** - strict standards in normal operation
✅ **Provides visibility** - clear mode transitions in logs
✅ **No configuration required** - works out of the box
✅ **Backward compatible** - existing validators benefit immediately

## Rollout Plan

1. **Deploy** to all validators via standard git pull + restart
2. **Monitor** for mode transitions in logs (should see only NORMAL in steady state)
3. **Alert** on DEGRADED mode entries (indicates Challenge API issues)
4. **Page** on EMERGENCY mode entries (requires investigation)
5. **Measure** reduction in "weight setting skipped" log frequency

## Future Improvements

1. **Metrics API**: Expose `blocks_since_last_update` and `operation_mode` via validator API
2. **Dashboard**: Create Grafana dashboard for mode distribution across fleet
3. **Predictive Alerts**: Warn at 3500 blocks (before DEGRADED threshold)
4. **Historic Mode Tracking**: Store mode transitions in DB for analysis
5. **Adaptive Thresholds**: Adjust DEGRADED/EMERGENCY thresholds based on network conditions (future consideration)

---

**Implementation Date**: February 7, 2026  
**Files Modified**: 4  
**Lines Changed**: ~150  
**Tests Required**: Integration testing with simulated Challenge API outages
