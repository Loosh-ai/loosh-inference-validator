# Inference Response Evaluation Process

This document describes the evaluation pipeline for scoring and ranking responses from multiple miners.

## Overview

The evaluation process:
1. **Assesses response quality** (runs FIRST, before clustering)
2. Determines consensus among quality-filtered responses
3. Identifies outliers using quality-aware detection
4. Scores each individual response
5. Calculates emissions (rewards) for each miner
6. Generates visualizations and narrative summaries

## Step-by-Step Process

### 1. Embedding Generation

Each text response is converted to embeddings using a sentence transformer model (all-MiniLM-L6-v2). Cosine similarity is computed between all response pairs to create a similarity matrix.

### 2. Quality Assessment (First Pass)

**This step runs BEFORE clustering to prevent low-quality responses from forming consensus.**

Each response is evaluated on four semantic quality dimensions:

| Dimension | Description |
|-----------|-------------|
| **Prompt Relevance** | How well the response addresses the original prompt |
| **Information Density** | Richness of meaningful content |
| **Specificity** | Concrete details vs. vague generalities |
| **Coherence** | Logical flow and internal consistency |

Responses scoring below the quality threshold (0.35) are flagged as low-quality and removed from consensus consideration. This prevents "garbage consensus" attacks where coordinated low-quality responses could dominate.

### 3. Filtering and Quality Control

After quality assessment, additional filters are applied:

#### Smart Outlier Detection
Local Outlier Factor (LOF) identifies responses that are significantly different from others. **Quality-aware logic protects high-quality unique responses** from being incorrectly flagged as outliers - a high-quality response that differs from low-quality consensus is preserved.

#### Length-Based Quality Filtering
Responses shorter than a threshold (based on average length and quality sensitivity) are filtered out. Quality sensitivity is configurable (lower = stricter).

#### Clustering
Agglomerative clustering groups similar responses. The dominant cluster (largest group) is identified. **High-quality responses have more influence on cluster determination** (quality-weighted consensus).

### 4. Consensus Measurement

Pairwise cosine similarity is computed for all remaining responses. The consensus score combines:
- Mean similarity (μ)
- Standard deviation of similarities (σ)
- Final score: μ + λ·σ (where λ is a configurable factor, default 1.0)

Consensus is achieved if the score exceeds the threshold (default 0.7). If polarity clustering is enabled, responses must also agree on affirmative/negative classification.

**Garbage Detection Alerts:** When low-quality clusters are detected, automatic warnings are logged to alert operators of potential coordinated attacks.

### 5. Individual Response Scoring

Each response receives a composite score with the following weights:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Similarity** | 40% | Average similarity to other quality responses |
| **Quality** | 25% | Semantic quality score (relevance, density, specificity, coherence) |
| **Confidence** | 15% | Miner-provided confidence value (if available) |
| **Consensus** | 10% | Bonus for membership in consensus cluster |
| **Diversity** | 15% | Bonus for unique high-quality responses |

#### Diversity Bonus
High-quality responses that are unique (not part of the main cluster) can receive up to **+15% bonus**. This rewards miners who provide valuable alternative perspectives rather than copying others.

Out-of-consensus responses are scored based on their similarity to the consensus cluster, with a penalty applied (unless they qualify for diversity bonus).

### 6. Heatmap Generation

A similarity heatmap visualizes pairwise similarities between responses:
- Rows and columns represent responses
- Cell colors indicate similarity (warmer = more similar)
- Saved as PNG and uploaded to storage

### 7. Narrative Generation

An LLM generates a narrative summary that includes:
- Original prompt analysis
- Response comparison
- Evaluation methodology
- Consensus classification (in/out)
- Final consensus determination

The narrative generator uses consensus scores, heatmap paths, and response classifications as input.

### 8. Emissions Calculation

Emissions allocation considers:

#### Base Calculation
- Response speed: Faster responses receive higher base allocation (inverse time ratio)
- Consensus score: Higher overall consensus increases total emissions pool

#### Consensus Bonus
Responses in the consensus cluster receive a 1.2x multiplier.

#### Highest Score Bonus
The highest-scoring response receives a scaled bonus multiplier based on the absolute difference between the highest and second-highest scores:
- Small difference (~1 point): ~1.05x multiplier
- Medium difference (~10 points): ~1.24x multiplier  
- Large difference (~20+ points): up to 1.5x multiplier (capped)

The scaling uses a tanh function with scale factor 0.1 to map score differences to the [1.0, 1.5] multiplier range.

## Configuration Options

- **Clustering:** Enable/disable agglomerative clustering
- **Weighted Scoring:** Factor in confidence values when available
- **Outlier Detection:** Enable LOF-based outlier removal (now quality-aware)
- **Quality Filtering:** Filter responses below length threshold
- **Quality Sensitivity:** Quality filter strictness (0.0-1.0, lower = stricter)
- **Semantic Quality Threshold:** Minimum quality score (default: 0.35)
- **Consensus Threshold:** Minimum score for consensus (default: 0.7)
- **Lambda Factor:** Standard deviation multiplier in consensus calculation (default: 1.0)
- **Heatmap Generation:** Generate similarity visualization
- **Polarity Clustering:** Require agreement on affirmative/negative classification

## Outputs

1. **Consensus Score:** Overall agreement metric (0.0 to 1.0+)
2. **Heatmap Image:** Pairwise similarity visualization
3. **Narrative:** LLM-generated evaluation summary
4. **Individual Scores:** Per-response composite scores
5. **Emissions Allocation:** Reward distribution per miner
6. **In-Consensus List:** Responses in the consensus cluster
7. **Out-of-Consensus List:** Filtered or non-matching responses
8. **Quality Scores:** Per-response semantic quality assessments

## Result Interpretation

**High Consensus Score (>0.7):** Strong agreement among quality responses. In-consensus responses receive bonuses.

**Low Consensus Score (<0.7):** Weak or no consensus. May indicate ambiguous prompts, diverse valid responses, or attack detection (low-quality cluster filtered).

**Individual Scores:** Composite metric combining similarity, quality, confidence, consensus membership, and diversity. Highest-scoring response receives scaled emissions bonus.

**Emissions Distribution:** Allocated based on response speed, consensus participation, quality, and diversity. The highest-scoring response receives additional bonus proportional to score advantage over second-best.

---

## Weight Setting & On-Chain Emissions

After evaluation, emissions data is stored in the database and used to calculate on-chain weights that determine TAO rewards.

### Miner Filtering

Before calculating weights, the validator filters nodes to include only **serving miners**:

| Filter | Reason |
|--------|--------|
| Validator's own node | Self-exclusion |
| Stake >= 999 TAO | High stake indicates validator, not miner |
| No IP/port advertised | Node not serving |
| In validator database | Known validators from Challenge API |

This ensures weights are only set for active, serving miners.

### Freshness Gate

Miners without recent successful responses receive **zero weight**, preventing stale miners from accumulating weight based on old performance.

| Mode | Freshness Window | Triggered When |
|------|------------------|----------------|
| Normal | 3 hours | < 4000 blocks since last weight update |
| Degraded | 24 hours | 4000-4499 blocks since last update |
| Emergency | No limit | >= 4500 blocks since last update |

See [Tiered Fallback Strategy](#tiered-fallback-strategy) for details.

### EMA (Exponential Moving Average) Scoring

The validator uses **EMA scoring** to calculate miner weights, which provides a fair and stable scoring mechanism:

**What is EMA?**
EMA gives more weight to recent performance while still considering historical data, preventing:
- Single lucky responses from disproportionately rewarding miners
- Short-term fluctuations from destabilizing weights
- Gaming through occasional high-quality responses

**EMA Formula:**
```
EMA = alpha x current_emission + (1 - alpha) x previous_EMA
```

Where:
- **alpha** = 0.3 (default) - smoothing factor
  - Higher alpha = more weight to recent performance
  - Lower alpha = more weight to historical performance
- **lookback period** = 24 hours (default) - how far back to consider

**Example:** With alpha=0.3, a miner's score reflects:
- 30% from their most recent evaluation
- 70% carried over from their previous EMA score

This means consistent performers are rewarded over miners who occasionally perform well.

### Weight Setting Process

The validator sets weights on-chain every **72 minutes** (hard-coded for network consistency):

1. **Filter Serving Miners**: Exclude validators, high-stake nodes, and non-serving nodes
2. **Apply Freshness Gate**: Zero out stale miners based on current operation mode
3. **Calculate EMA Scores**: Query evaluation emissions and compute EMA for each miner
4. **Check for Zero Weights**: If all weights are zero, skip weight setting (see below)
5. **Normalize Weights**: Scale scores so they sum to 1.0 (required by Bittensor)
6. **Check Rate Limits**: Verify enough blocks have passed since last weight update
7. **Submit to Chain**: Set weights on-chain using the Bittensor SDK

**Zero Weight Behavior:** When all EMA scores are zero (after freshness gate), weight setting is **skipped entirely** rather than distributing weights uniformly. This prevents rewarding untested, down, or adversarial miners during:
- Validator startup (no evaluations yet)
- Database outages
- All miners being stale

**Technical Note:** The validator uses the **Bittensor SDK v10+** for weight setting instead of Fiber because:
- Commit Reveal v3 (CRv3) has been removed from the chain
- The SDK automatically handles CRv4 commit-reveal at the chain level
- See: [Bittensor SDK v10 Migration Guide](https://docs.learnbittensor.org/sdk/migration-guide)

### Tiered Fallback Strategy

Validators face a critical challenge: if they can't set weights for ~5000 blocks, they risk deregistration. The tiered fallback strategy balances quality standards against deregistration risk.

#### Normal Mode (< 4000 blocks since last update)
- **Freshness gate:** 3 hours
- **Behavior:** Standard operation - skip weight setting if all weights zero
- **Rationale:** Plenty of time before deregistration; maintain strict quality

#### Degraded Mode (4000-4499 blocks since last update)
- **Freshness gate:** 24 hours (relaxed from 3 hours)
- **Behavior:** Uses emissions from local validator DB (stored during evaluation)
- **Alerts:** `DEGRADED MODE` warnings logged
- **Rationale:** Challenge API may be down, but local evaluations are still valid

#### Emergency Mode (>= 4500 blocks since last update)
- **Freshness gate:** None - uses ALL available local emissions
- **Behavior:** If still no emissions, distributes minimal uniform weights as last resort
- **Alerts:** `EMERGENCY MODE` critical alerts
- **Rationale:** Prevent deregistration at all costs; operator intervention urgently needed

**Why This Doesn't Skew Weights:**
- Local emissions are real (same consensus algorithm used regardless of Challenge API)
- Degraded/Emergency modes use actual performance data from local evaluations
- Uniform distribution only happens in extreme edge case (no local DB emissions + imminent deregistration)

### How This Affects Miner Rewards

1. **Every evaluation** generates emissions data stored in the database
2. **EMA scores** are calculated from emissions over the lookback period (24h default)
3. **Weights are set** on-chain every 72 minutes
4. **TAO rewards** are distributed based on weights each epoch (~12 seconds)

**For miners, this means:**
- **Consistency matters**: Steady performance builds a higher EMA over time
- **Recovery is gradual**: A single bad response won't tank your rewards immediately, but consistently poor performance will lower your EMA
- **Gaming is difficult**: You can't benefit from occasional bursts of performance
- **Stay online**: The freshness gate requires responses within 3 hours to receive weight
- **Quality matters**: Low-quality responses are filtered before consensus, reducing their impact

### Database Cleanup

To prevent unbounded database growth while maintaining sufficient history for EMA calculation:

- **Retention period**: 48 hours (2x the EMA lookback window)
- **Cleaned data**: Challenges, responses, evaluation results, miner scores
- **Preserved data**: Sybil detection records (managed separately by SybilSyncTask)
- **Cleanup frequency**: Every 24 hours (configurable)

This ensures the database stays manageable while always having enough historical data for accurate EMA calculations.

### Configuration Options

All operational parameters below are defined in `validator/internal_config.py` and are NOT configurable via environment variables. This ensures network consistency across all validators.

#### Miner Selection (Internal)

| Parameter | Default | Description |
|-----------|---------|-------------|
| MIN_MINERS | 3 | Minimum miners for valid challenge round |
| MAX_MINERS | 10 | Maximum miners to query per challenge |
| MIN_STAKE_THRESHOLD | 100 TAO | Minimum stake for miner eligibility |
| MAX_MINER_STAKE | 999 TAO | Maximum stake to be considered a miner |

#### Challenge Timing (Internal)

| Parameter | Default | Description |
|-----------|---------|-------------|
| CHALLENGE_INTERVAL_SECONDS | 300 | Time between challenge cycles (5 min) |
| CHALLENGE_TIMEOUT_SECONDS | 120 | Timeout for miner responses (2 min) |
| EVALUATION_TIMEOUT_SECONDS | 300 | Timeout for evaluation (5 min) |

#### Scoring (Internal)

| Parameter | Default | Description |
|-----------|---------|-------------|
| SCORE_THRESHOLD | 0.7 | Minimum score for valid responses |

#### Weight Setting (Internal)

| Parameter | Default | Description |
|-----------|---------|-------------|
| WEIGHTS_INTERVAL_SECONDS | 4320 | Weight update interval (72 min) |
| WEIGHT_MIN_SERVING_NODES | 1 | Minimum serving miners to set weights |
| WEIGHT_FRESHNESS_HOURS | 3 | Stale threshold in normal mode |
| WEIGHT_FRESHNESS_HOURS_DEGRADED | 24 | Stale threshold in degraded mode |

#### Deregistration Safety (Internal)

| Parameter | Default | Description |
|-----------|---------|-------------|
| DEREGISTRATION_BLOCK_LIMIT | 5000 | Blocks until deregistration |
| DEGRADED_MODE_THRESHOLD | 4000 | Blocks to trigger degraded mode |
| EMERGENCY_MODE_THRESHOLD | 4500 | Blocks to trigger emergency mode |

#### Configurable via Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| EMA Alpha | 0.3 | Smoothing factor (higher = more recent weight) |
| EMA Lookback | 24 hours | Historical period for EMA calculation |
| DB Retention | 48 hours | How long to keep evaluation data |
| DB Cleanup Interval | 24 hours | How often to run database cleanup |

**Important:** To modify internal parameters, edit `validator/internal_config.py` directly and redeploy. Do not create environment variables for these - they will be ignored.
