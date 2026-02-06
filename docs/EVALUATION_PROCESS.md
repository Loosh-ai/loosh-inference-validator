# Inference Response Evaluation Process

This document describes the evaluation pipeline for scoring and ranking responses from multiple miners.

## Overview

The evaluation process:
1. Determines consensus among responses
2. Identifies outliers and low-quality responses
3. Scores each individual response
4. Calculates emissions (rewards) for each miner
5. Generates visualizations and narrative summaries

## Step-by-Step Process

### 1. Embedding Generation

Each text response is converted to embeddings using a sentence transformer model (all-MiniLM-L6-v2). Cosine similarity is computed between all response pairs to create a similarity matrix.

### 2. Filtering and Quality Control

The system applies filters before consensus evaluation:

#### Outlier Detection
Local Outlier Factor (LOF) identifies responses that are significantly different from others. Outliers are removed from consensus evaluation.

#### Quality Filtering
Responses shorter than a threshold (based on average length and quality sensitivity) are filtered out. Quality sensitivity is configurable (lower = stricter).

#### Clustering
Agglomerative clustering groups similar responses. The dominant cluster (largest group) is identified, and responses outside this cluster are excluded from consensus evaluation.

### 3. Consensus Measurement

Pairwise cosine similarity is computed for all remaining responses. The consensus score combines:
- Mean similarity (μ)
- Standard deviation of similarities (σ)
- Final score: μ + λ·σ (where λ is a configurable factor, default 1.0)

Consensus is achieved if the score exceeds the threshold (default 0.7). If polarity clustering is enabled, responses must also agree on affirmative/negative classification.

### 4. Individual Response Scoring

Each response receives a composite score:

#### Consensus Alignment (50% weight)
Average similarity to all other responses in the filtered set.

#### Quality Score (20% weight)
Normalized response length (word count relative to maximum length).

#### Confidence Score (20% weight)
If confidence values are provided and weighted scoring is enabled, confidence is factored in.

#### Consensus Bonus (10% weight)
Fixed bonus for responses in the consensus cluster.

Out-of-consensus responses are scored based on their similarity to the consensus cluster, with a penalty applied.

### 5. Heatmap Generation

A similarity heatmap visualizes pairwise similarities between responses:
- Rows and columns represent responses
- Cell colors indicate similarity (warmer = more similar)
- Saved as PNG and uploaded to storage

### 6. Narrative Generation

An LLM generates a narrative summary that includes:
- Original prompt analysis
- Response comparison
- Evaluation methodology
- Consensus classification (in/out)
- Final consensus determination

The narrative generator uses consensus scores, heatmap paths, and response classifications as input.

### 7. Emissions Calculation

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
- **Outlier Detection:** Enable LOF-based outlier removal
- **Quality Filtering:** Filter responses below length threshold
- **Quality Sensitivity:** Quality filter strictness (0.0-1.0, lower = stricter)
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

## Result Interpretation

**High Consensus Score (>0.7):** Strong agreement among responses. In-consensus responses receive bonuses.

**Low Consensus Score (<0.7):** Weak or no consensus. May indicate ambiguous prompts or low-quality responses.

**Individual Scores:** Composite metric combining alignment, quality, confidence, and consensus membership. Highest-scoring response receives scaled emissions bonus.

**Emissions Distribution:** Allocated based on response speed, consensus participation, and individual quality. The highest-scoring response receives additional bonus proportional to score advantage over second-best.

## Weight Setting & On-Chain Emissions

After evaluation, emissions data is stored in the database and used to calculate on-chain weights that determine TAO rewards.

### EMA (Exponential Moving Average) Scoring

The validator uses **EMA scoring** to calculate miner weights, which provides a fair and stable scoring mechanism:

**What is EMA?**
EMA gives more weight to recent performance while still considering historical data, preventing:
- Single lucky responses from disproportionately rewarding miners
- Short-term fluctuations from destabilizing weights
- Gaming through occasional high-quality responses

**EMA Formula:**
```
EMA = α × current_emission + (1 - α) × previous_EMA
```

Where:
- **α (alpha)** = 0.3 (default) - smoothing factor
  - Higher α = more weight to recent performance
  - Lower α = more weight to historical performance
- **lookback period** = 24 hours (default) - how far back to consider

**Example:** With α=0.3, a miner's score reflects:
- 30% from their most recent evaluation
- 70% carried over from their previous EMA score

This means consistent performers are rewarded over miners who occasionally perform well.

### Weight Setting Process

The validator periodically sets weights on-chain (default: every 30 minutes):

1. **Calculate EMA Scores**: Query all evaluation emissions from the last 24 hours and compute EMA for each miner
2. **Normalize Weights**: Scale scores so they sum to 1.0 (required by Bittensor)
3. **Check Rate Limits**: Verify enough blocks have passed since last weight update
4. **Submit to Chain**: Set weights on-chain using the Bittensor SDK

**Technical Note:** The validator uses the **Bittensor SDK v10+** for weight setting instead of Fiber because:
- Commit Reveal v3 (CRv3) has been removed from the chain
- The SDK automatically handles CRv4 commit-reveal at the chain level
- See: [Bittensor SDK v10 Migration Guide](https://docs.learnbittensor.org/sdk/migration-guide)

### How This Affects Miner Rewards

1. **Every evaluation** generates emissions data stored in the database
2. **EMA scores** are calculated from emissions over the lookback period (24h default)
3. **Weights are set** on-chain periodically (30 min default)
4. **TAO rewards** are distributed based on weights each epoch (~12 seconds)

**For miners, this means:**
- **Consistency matters**: Steady performance builds a higher EMA over time
- **Recovery is gradual**: A single bad response won't tank your rewards immediately, but consistently poor performance will lower your EMA
- **Gaming is difficult**: You can't benefit from occasional bursts of performance

### Database Cleanup

To prevent unbounded database growth while maintaining sufficient history for EMA calculation:

- **Retention period**: 48 hours (2x the EMA lookback window)
- **Cleaned data**: Challenges, responses, evaluation results, miner scores
- **Preserved data**: Sybil detection records (managed separately by SybilSyncTask)
- **Cleanup frequency**: Every 24 hours (configurable)

This ensures the database stays manageable while always having enough historical data for accurate EMA calculations.

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WEIGHTS_INTERVAL_SECONDS` | 1800 (30 min) | How often to set weights on-chain |
| EMA Alpha | 0.3 | Smoothing factor (higher = more recent weight) |
| EMA Lookback | 24 hours | Historical period for EMA calculation |
| DB Retention | 48 hours | How long to keep evaluation data |
| DB Cleanup Interval | 24 hours | How often to run database cleanup |