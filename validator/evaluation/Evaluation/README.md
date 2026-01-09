# Evaluation Tools

Advanced consensus evaluation engine for analyzing AI model response alignment and quality.

## 📁 Contents

### `consensus_engine.py`
Sophisticated consensus evaluation system with multiple scoring methods, clustering, and visualization capabilities.

## 🔧 Consensus Engine

A comprehensive system for evaluating consensus among AI model responses using multiple analytical approaches including similarity analysis, clustering, outlier detection, and quality filtering.

### Features

- **Multi-Modal Scoring**: Basic similarity, weighted confidence, and polarity clustering
- **Advanced Filtering**: Outlier detection, quality-based filtering, and clustering analysis
- **Visualization**: Similarity heatmaps and quality distribution plots
- **Configurable Pipeline**: Flexible evaluation configuration with multiple methods
- **Statistical Analysis**: Mean, standard deviation, and threshold-based consensus determination

### Core Components

#### ConsensusConfig

Comprehensive configuration for evaluation parameters:

```python
@dataclass
class ConsensusConfig:
    use_clustering: bool = False              # Enable agglomerative clustering
    use_weighted_scoring: bool = False        # Enable confidence weighting
    use_polarity_clustering: bool = False     # Enable polarity-based analysis
    use_outlier_detection: bool = False       # Enable LOF outlier detection
    apply_quality_filter: bool = False        # Enable response quality filtering
    quality_sensitivity: float = 0.5          # Quality filter sensitivity (0-1)
    generate_heatmap: bool = False            # Generate similarity heatmap
    heatmap_path: str = "./output/..."        # Heatmap output path
    lambda_factor: float = 1.0                # Similarity score adjustment factor
    threshold_min: float = 0.7                # Minimum consensus threshold
    polarity_agreement_min: float = 0.8       # Minimum polarity agreement
```

#### ConsensusResult

Comprehensive results object containing all evaluation metrics:

```python
@dataclass
class ConsensusResult:
    original_prompt: str                      # Original evaluation prompt
    in_consensus: Dict[str, str]             # Responses achieving consensus
    out_of_consensus: Dict[str, str]         # Responses outside consensus
    similarity_score: float                  # θ = μ + λ·σ consensus score
    weighted_score: Optional[float]          # Confidence-weighted score
    polarity_agreement: Optional[float]      # Polarity clustering score
    heatmap_path: Optional[str]             # Generated heatmap path
    consensus_achieved: bool                 # Final consensus determination
    quality_plot_path: Optional[str]        # Quality analysis plot path
    consensus_narrative: Optional[str]       # Generated narrative summary
    miner_scores: Optional[Dict[str, float]] # Individual response scores
```

### Usage

#### Basic Consensus Evaluation

```python
from Evaluation.consensus_engine import (
    ConsensusEngine, ConsensusConfig
)
import numpy as np

# Prepare evaluation data
prompt = "What is the best approach to climate change mitigation?"
responses = [
    "Focus on renewable energy transition and carbon pricing",
    "Implement strict regulations and international cooperation", 
    "Invest in technology innovation and green infrastructure",
    "Promote individual responsibility and lifestyle changes"
]
embeddings = [get_embedding(response) for response in responses]

# Initialize consensus engine
engine = ConsensusEngine(
    original_prompt=prompt,
    responses=responses,
    embeddings=embeddings
)

# Configure evaluation
config = ConsensusConfig(
    threshold_min=0.7,
    lambda_factor=1.0,
    generate_heatmap=True
)

# Evaluate consensus
result = engine.evaluate_consensus(config)

print(f"Consensus achieved: {result.consensus_achieved}")
print(f"Similarity score: {result.similarity_score:.3f}")
print(f"Responses in consensus: {len(result.in_consensus)}")
```

#### Advanced Configuration

```python
# Advanced evaluation with all features enabled
advanced_config = ConsensusConfig(
    use_clustering=True,                    # Group similar responses
    use_weighted_scoring=True,              # Weight by confidence
    use_polarity_clustering=True,           # Analyze agreement polarity
    use_outlier_detection=True,             # Remove outlier responses
    apply_quality_filter=True,              # Filter low-quality responses
    quality_sensitivity=0.3,                # Strict quality filtering
    generate_heatmap=True,                  # Generate visualizations
    lambda_factor=1.5,                      # Adjust consensus sensitivity
    threshold_min=0.8,                      # High consensus threshold
    polarity_agreement_min=0.85             # Strong polarity agreement required
)

# Include confidence scores and polarity labels
confidences = [0.9, 0.7, 0.8, 0.6]
polarities = ["affirmative", "affirmative", "affirmative", "negative"]

engine = ConsensusEngine(
    original_prompt=prompt,
    responses=responses,
    embeddings=embeddings,
    confidences=confidences,
    polarities=polarities
)

result = engine.evaluate_consensus(advanced_config)
```

### Evaluation Methods

#### 1. Basic Similarity Scoring

Calculates consensus using cosine similarity:
```
θ = μ + λ·σ
where μ = mean similarity, σ = standard deviation, λ = adjustment factor
```

#### 2. Weighted Scoring

Incorporates confidence scores:
```
Score = Σ(similarity_ij × confidence_i × confidence_j) / Σ(confidence_i × confidence_j)
```

#### 3. Polarity Clustering

Analyzes agreement based on response sentiment:
```
Agreement = max(polarity_group_count) / total_responses
```

#### 4. Outlier Detection

Uses Local Outlier Factor (LOF) to identify anomalous responses:
```python
lof = LocalOutlierFactor(n_neighbors=2)
outliers = lof.fit_predict(embeddings) == -1
```

#### 5. Quality Filtering

Filters responses based on length and content quality:
```python
min_length = average_length × sensitivity
quality_mask = [len(response.split()) >= min_length for response in responses]
```

#### 6. Agglomerative Clustering

Groups similar responses and selects dominant cluster:
```python
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.5,
    affinity='precomputed'
)
```

### Visualization

#### Similarity Heatmap

Generates annotated heatmap showing pairwise similarities:

```python
config = ConsensusConfig(
    generate_heatmap=True,
    heatmap_path="./output/consensus_heatmap.png"
)

result = engine.evaluate_consensus(config)
# Heatmap saved to specified path
```

#### Quality Distribution Plot

Shows response length distribution and quality filtering impact:

```python
config = ConsensusConfig(
    apply_quality_filter=True,
    quality_sensitivity=0.5,
    generate_heatmap=True  # Also generates quality plot
)
```

### Performance Optimization

#### Batch Processing

```python
def evaluate_batch_consensus(prompts: List[str], 
                           response_sets: List[List[str]],
                           embedding_sets: List[List[np.ndarray]],
                           config: ConsensusConfig) -> List[ConsensusResult]:
    results = []
    for prompt, responses, embeddings in zip(prompts, response_sets, embedding_sets):
        engine = ConsensusEngine(prompt, responses, embeddings)
        result = engine.evaluate_consensus(config)
        results.append(result)
    return results
```

#### Memory Optimization

```python
# For large-scale evaluation, process in chunks
def evaluate_large_dataset(dataset, chunk_size=100):
    results = []
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        chunk_results = evaluate_batch_consensus(chunk)
        results.extend(chunk_results)
        # Clear intermediate results to free memory
        del chunk_results
    return results
```

### Integration Examples

#### With Recording Tools

```python
from Recording.consensus_narrative_generator import ConsensusNarrativeGenerator

# Evaluate consensus and generate narrative
result = engine.evaluate_consensus(config)

# Generate human-readable summary
generator = ConsensusNarrativeGenerator(llm_config)
narrative = generator.generate_narrative(result)

# Complete evaluation report
print(f"Consensus Score: {result.similarity_score:.3f}")
print(f"Consensus Achieved: {result.consensus_achieved}")
print(f"\nNarrative Summary:\n{narrative}")
```

#### With Settings Management

```python
from loosh_utilities.Settings import LooshMemorySettings

settings = LooshMemorySettings.from_env()

config = ConsensusConfig(
    threshold_min=settings.SIMILARITY_THRESHOLD,
    generate_heatmap=True,
    heatmap_path=f"{settings.LOG_FILE.parent}/consensus_heatmap.png"
)
```

### Statistical Analysis

The consensus engine provides detailed statistical metrics:

```python
result = engine.evaluate_consensus(config)

print(f"Statistical Analysis:")
print(f"- Similarity Score (θ): {result.similarity_score:.4f}")
print(f"- Weighted Score: {result.weighted_score:.4f}")
print(f"- Polarity Agreement: {result.polarity_agreement:.2f}")
print(f"- Responses in Consensus: {len(result.in_consensus)}/{len(responses)}")
print(f"- Consensus Threshold Met: {result.consensus_achieved}")
```

## 🚀 Advanced Features

### Custom Similarity Metrics

Extend the engine with custom similarity calculations:

```python
class CustomConsensusEngine(ConsensusEngine):
    def _compute_pairwise_similarity(self) -> np.ndarray:
        # Custom similarity computation
        return custom_similarity_matrix(self.embeddings)
```

### Dynamic Threshold Adjustment

Implement adaptive thresholding based on dataset characteristics:

```python
def dynamic_threshold(responses: List[str]) -> float:
    complexity = np.mean([len(r.split()) for r in responses])
    return 0.6 + (complexity / 100) * 0.2  # Adjust based on complexity
```

## 🔍 Debugging and Analysis

### Detailed Diagnostics

```python
# Enable comprehensive analysis
config = ConsensusConfig(
    use_clustering=True,
    use_weighted_scoring=True,
    use_outlier_detection=True,
    apply_quality_filter=True,
    generate_heatmap=True
)

result = engine.evaluate_consensus(config)

# Analyze filtering impact
original_count = len(engine.all_responses)
final_count = len(result.in_consensus) + len(result.out_of_consensus)
filtered_count = original_count - final_count

print(f"Filtering Analysis:")
print(f"- Original responses: {original_count}")
print(f"- After filtering: {final_count}")
print(f"- Filtered out: {filtered_count}")
```

## 📋 Dependencies

- **numpy**: Numerical computations and array operations
- **scikit-learn**: Clustering, similarity metrics, and outlier detection
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **dataclasses**: Structured configuration and results

## 🔗 Related Components

- **Recording**: Narrative generation from consensus results
- **Settings**: Configuration management for evaluation parameters
- **Scoring**: Performance metrics and optimization integration 