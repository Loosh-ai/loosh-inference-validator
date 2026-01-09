from typing import Dict, Optional
from dataclasses import dataclass
import os

from .llm_service import LLMService, LLMConfig as LLMServiceConfig


@dataclass
class LLMConfig:
    """Legacy config for backward compatibility - use LLMServiceConfig instead."""
    api_url: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 800
    api_key: Optional[str] = None


@dataclass
class ConsensusResult:
    original_prompt: str
    in_consensus: Dict[str, str]
    out_of_consensus: Dict[str, str]
    similarity_score: float
    weighted_score: Optional[float]
    polarity_agreement: Optional[float]
    heatmap_path: Optional[str]
    consensus_achieved: bool
    quality_plot_path: Optional[str] = None  # NEW: add quality visual reference


class ConsensusNarrativeGenerator:
    def __init__(self, llm_config: LLMConfig, llm_service: Optional[LLMService] = None):
        """
        Initialize the consensus narrative generator.
        
        Args:
            llm_config: Legacy LLM configuration (for backward compatibility)
            llm_service: Optional LLMService instance. If not provided, will create one.
        """
        self.llm_config = llm_config
        self._llm_service = llm_service
        self._llm_name = "consensus-narrative-generator"
        self._initialized = False

    async def _ensure_llm_service(self) -> LLMService:
        """Ensure LLM service is initialized and configured."""
        if self._llm_service is None:
            self._llm_service = LLMService()
            await self._llm_service.initialize()
        
        if not self._initialized:
            # Determine provider from API URL
            api_url = self.llm_config.api_url.lower()
            if "azure" in api_url or "openai.azure.com" in api_url:
                provider = "azure_openai"
                # For Azure, use the full URL as api_base
                api_base = self.llm_config.api_url
            elif "anthropic" in api_url or "claude" in api_url:
                provider = "anthropic"
                # For Anthropic, extract base URL (remove /v1/messages if present)
                api_base = self.llm_config.api_url.replace("/v1/messages", "").rstrip("/")
            elif "ollama" in api_url:
                provider = "ollama"
                # For Ollama, use the base URL
                api_base = self.llm_config.api_url.replace("/api/chat", "").replace("/api/generate", "").rstrip("/")
            else:
                # Default to OpenAI
                provider = "openai"
                # For OpenAI, extract base URL (remove /v1/chat/completions if present)
                api_base = self.llm_config.api_url.replace("/v1/chat/completions", "").rstrip("/")
                # If it's still the default OpenAI URL, set to None (uses default)
                if api_base == "https://api.openai.com":
                    api_base = None
            
            # Get API key from config or environment
            api_key = self.llm_config.api_key or os.getenv("OPENAI_API_KEY")
            
            # Create LLM service config
            service_config = LLMServiceConfig(
                provider=provider,
                model=self.llm_config.model_name,
                api_key=api_key,
                api_base=api_base,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                timeout=60,
                max_retries=2
            )
            
            # Register the LLM
            await self._llm_service.register_llm(self._llm_name, service_config)
            self._initialized = True
        
        return self._llm_service

    def _construct_prompt(self, result: ConsensusResult) -> str:
        in_consensus_str = "\n".join([f"{k}: {v}" for k, v in result.in_consensus.items()])
        out_of_consensus_str = "\n".join([f"{k}: {v}" for k, v in result.out_of_consensus.items()]) or "None"

        scoring_lines = [
            f"- Basic Similarity Score (μ + λ·σ): {result.similarity_score:.4f}"
        ]
        if result.weighted_score is not None:
            scoring_lines.append(f"- Weighted Score: {result.weighted_score:.4f}")
        if result.polarity_agreement is not None:
            scoring_lines.append(f"- Polarity Agreement: {result.polarity_agreement:.2f}")
        if result.heatmap_path:
            scoring_lines.append(f"- Heatmap File: {result.heatmap_path}")
        if result.quality_plot_path:
            scoring_lines.append(f"- Quality Plot: {result.quality_plot_path} (indicates response length variance)")

        scoring_summary = "\n".join(scoring_lines)
        consensus_statement = "✅ Consensus was achieved." if result.consensus_achieved else "❌ Consensus was not achieved."

        return f"""
You are an expert analyst. Given an original prompt, a set of AI-generated responses, and the results of a consensus evaluation,
write an insightful narrative covering:

1. What the prompt was asking.
2. The nature of the responses and how they compare.
3. The process and metrics used to evaluate consensus.
4. Any quality filtering that was applied and its impact.
5. A summary of which responses were in consensus and which were not.
6. The final consensus result.

---

### Original Prompt
{result.original_prompt}

---

### In-Consensus Responses
{in_consensus_str}

---

### Out-of-Consensus Responses
{out_of_consensus_str}

---

### Scoring Summary
{scoring_summary}

---

### Final Verdict
{consensus_statement}

Please write a concise but insightful narrative summary.
""".strip()

    async def generate_narrative(self, result: ConsensusResult) -> str:
        """
        Generate a narrative from consensus results using the LLM service.
        
        Args:
            result: ConsensusResult containing evaluation data
            
        Returns:
            Generated narrative string
        """
        prompt = self._construct_prompt(result)
        
        # Ensure LLM service is initialized
        llm_service = await self._ensure_llm_service()
        
        # Prepare messages in OpenAI format
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes analytical summaries."},
            {"role": "user", "content": prompt}
        ]
        
        # Invoke LLM service
        response = await llm_service.invoke(
            name=self._llm_name,
            messages=messages,
            timeout=60.0
        )
        
        return response.content.strip()
