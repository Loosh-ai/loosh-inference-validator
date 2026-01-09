# Recording Tools

Tools for recording, analyzing, and generating narratives from AI model consensus evaluation results.

## 📁 Contents

### `consensus_narrative_generator.py`
Automated narrative generation from consensus evaluation results using LLM integration.

### `similarity_heatmap.py`
Visualization tools for consensus similarity analysis and heatmap generation.

## 🔧 Consensus Narrative Generator

Transforms consensus evaluation results into human-readable analytical narratives using configurable LLM backends.

### Features

- **Multi-LLM Support**: Compatible with various LLM APIs (OpenAI, local models, etc.)
- **Rich Context Integration**: Incorporates all consensus metrics and visualizations
- **Quality Assessment**: Includes response length variance and filtering analysis
- **Structured Narratives**: Produces consistent, insightful analytical summaries
- **Configurable Generation**: Adjustable temperature, tokens, and model parameters

### Usage

```python
from Recording.consensus_narrative_generator import (
    ConsensusNarrativeGenerator, LLMConfig, ConsensusResult
)

# Configure LLM backend
llm_config = LLMConfig(
    api_url="https://api.openai.com/v1/chat/completions",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=800
)

# Initialize generator
generator = ConsensusNarrativeGenerator(llm_config)

# Generate narrative from consensus results
narrative = generator.generate_narrative(consensus_result)
print(narrative)
```

### LLM Configuration

The `LLMConfig` dataclass supports flexible LLM integration:

```python
@dataclass
class LLMConfig:
    api_url: str          # LLM API endpoint
    model_name: str       # Model identifier
    temperature: float    # Generation randomness (0.0-1.0)
    max_tokens: int       # Maximum response length
```

**Supported Configurations:**

```python
# OpenAI GPT-4
openai_config = LLMConfig(
    api_url="https://api.openai.com/v1/chat/completions",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=800
)

# Local LLM (llama.cpp server)
local_config = LLMConfig(
    api_url="http://localhost:8080/v1/chat/completions",
    model_name="llama-2-7b-chat",
    temperature=0.5,
    max_tokens=600
)

# Anthropic Claude
claude_config = LLMConfig(
    api_url="https://api.anthropic.com/v1/messages",
    model_name="claude-3-sonnet-20240229",
    temperature=0.6,
    max_tokens=700
)
```

### Narrative Structure

Generated narratives follow a consistent analytical structure:

1. **Prompt Analysis**: What the original prompt was asking
2. **Response Comparison**: Nature and differences between AI responses
3. **Consensus Process**: Methods and metrics used for evaluation
4. **Quality Filtering**: Impact of quality controls and filtering
5. **Consensus Summary**: Which responses achieved consensus
6. **Final Assessment**: Overall consensus determination and insights

### Integration with Consensus Results

The generator works with `ConsensusResult` objects from the evaluation engine:

```python
@dataclass
class ConsensusResult:
    original_prompt: str                    # Original evaluation prompt
    in_consensus: Dict[str, str]           # Responses that achieved consensus
    out_of_consensus: Dict[str, str]       # Responses outside consensus
    similarity_score: float                # Basic similarity score
    weighted_score: Optional[float]        # Confidence-weighted score
    polarity_agreement: Optional[float]    # Polarity-based agreement
    heatmap_path: Optional[str]           # Path to similarity heatmap
    consensus_achieved: bool               # Final consensus determination
    quality_plot_path: Optional[str]      # Quality analysis visualization
```

### Example Output

```
Consensus Analysis: Multi-Agent Response Evaluation

The original prompt requested analysis of ethical decision-making in autonomous 
vehicle scenarios, specifically addressing the trolley problem in real-world 
contexts.

Response Characteristics:
The evaluation included 5 AI-generated responses, with 3 achieving consensus 
(R1, R3, R5) and 2 falling outside the consensus threshold (R2, R4). The 
in-consensus responses demonstrated strong alignment on utilitarian principles, 
while outlier responses showed deontological reasoning patterns.

Consensus Methodology:
The evaluation employed cosine similarity analysis with a threshold of 0.7, 
supplemented by confidence weighting (0.82) and polarity clustering (0.89). 
Quality filtering removed responses below 50% of average length, improving 
overall coherence.

Final Assessment:
✅ Consensus was achieved with a similarity score of 0.847, indicating strong 
alignment among the majority of responses on the ethical framework and 
decision criteria for autonomous vehicle scenarios.
```

## 📊 Similarity Heatmap

Visualization component for consensus similarity analysis (referenced in the JSON pipeline).

### Features

- **Interactive Heatmaps**: Visual similarity matrices for response analysis
- **Quality Distributions**: Response length and quality variance visualization
- **Consensus Clustering**: Visual representation of consensus groupings
- **Export Capabilities**: PNG, SVG, and interactive HTML outputs

### Integration

The similarity heatmap integrates with both the narrative generator and evaluation engine:

```python
# Heatmap generation is triggered by consensus evaluation
config = ConsensusConfig(
    generate_heatmap=True,
    heatmap_path="./output/consensus_similarity_heatmap.png"
)

result = consensus_engine.evaluate_consensus(config)

# Heatmap path is included in narrative generation
generator = ConsensusNarrativeGenerator(llm_config)
narrative = generator.generate_narrative(result)
# Narrative will reference: "Heatmap File: ./output/consensus_similarity_heatmap.png"
```

## 🚀 Advanced Usage

### Custom Prompt Templates

The narrative generator uses structured prompts that can be customized:

```python
class CustomNarrativeGenerator(ConsensusNarrativeGenerator):
    def _construct_prompt(self, result: ConsensusResult) -> str:
        # Custom prompt construction logic
        return custom_prompt_template.format(
            prompt=result.original_prompt,
            responses=result.in_consensus,
            metrics=self._format_metrics(result)
        )
```

### Batch Processing

Process multiple consensus results:

```python
def generate_batch_narratives(results: List[ConsensusResult], 
                            llm_config: LLMConfig) -> List[str]:
    generator = ConsensusNarrativeGenerator(llm_config)
    narratives = []
    
    for result in results:
        try:
            narrative = generator.generate_narrative(result)
            narratives.append(narrative)
        except Exception as e:
            narratives.append(f"Error generating narrative: {e}")
    
    return narratives
```

### Integration with Evaluation Pipeline

```python
from Evaluation.consensus_engine import ConsensusEngine
from Recording.consensus_narrative_generator import ConsensusNarrativeGenerator

# Complete evaluation and recording pipeline
def evaluate_and_record(prompt: str, responses: List[str], 
                       embeddings: List[np.ndarray]) -> str:
    # Evaluate consensus
    engine = ConsensusEngine(prompt, responses, embeddings)
    result = engine.evaluate_consensus(config)
    
    # Generate narrative
    generator = ConsensusNarrativeGenerator(llm_config)
    narrative = generator.generate_narrative(result)
    
    return narrative
```

## 🔍 Error Handling

The generator includes comprehensive error handling:

```python
try:
    narrative = generator.generate_narrative(result)
except requests.HTTPError as e:
    # Handle API errors
    logger.error(f"LLM API error: {e}")
except ValueError as e:
    # Handle configuration errors
    logger.error(f"Configuration error: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

## 📋 Dependencies

- **requests**: HTTP client for LLM API communication
- **dataclasses**: Structured configuration management
- **typing**: Type hints and annotations

## 🔗 Related Components

- **Evaluation**: Provides `ConsensusResult` objects for narrative generation
- **Settings**: Configuration management for LLM credentials and endpoints
- **n8n**: Integration with workflow automation pipelines 