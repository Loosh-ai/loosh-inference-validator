"""
LLM service for managing and invoking Large Language Models.

This service supports multiple chat model providers (ChatOpenAI, ChatAnthropic, etc.)
and allows for dynamic registration and selection of different model types.

This is a standalone utility service that can be used across Loosh projects.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, Callable
import structlog
from pydantic import BaseModel, Field
import asyncio
import hashlib
import json
import time
from diskcache import Cache

logger = structlog.get_logger(__name__)


class LLMConfig(BaseModel):
    """Configuration for an LLM instance."""
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic', 'ollama')")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_base: Optional[str] = Field(None, description="Custom API endpoint URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    streaming: bool = Field(default=False, description="Enable streaming responses")
    provider_specific_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")


class LLMResponse(BaseModel):
    """Response from an LLM invocation."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model that generated the response")
    provider: str = Field(..., description="Provider that generated the response")
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    response_time: float = Field(0.0, description="Response time in seconds")
    cached: bool = Field(False, description="Whether response was cached")


class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, retryable: bool = True):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


class ChatModelProvider(ABC):
    """
    Abstract base class for chat model providers.
    
    Each provider (OpenAI, Anthropic, Ollama, etc.) implements this interface
    to provide a consistent way to create and configure chat models.
    """
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider (e.g., 'openai', 'anthropic', 'ollama')."""
        pass
    
    @abstractmethod
    def create_chat_model(self, config: LLMConfig) -> Any:
        """
        Create a chat model instance from the configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            Chat model instance (e.g., ChatOpenAI, ChatAnthropic, ChatOllama)
        """
        pass
    
    @abstractmethod
    def get_required_dependencies(self) -> List[str]:
        """
        Get list of required package dependencies for this provider.
        
        Returns:
            List of package names (e.g., ['langchain-openai'])
        """
        pass
    
    def validate_config(self, config: LLMConfig) -> None:
        """
        Validate the configuration for this provider.
        
        Args:
            config: LLM configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if config.provider != self.get_provider_name():
            raise ValueError(
                f"Provider mismatch: expected '{self.get_provider_name()}', "
                f"got '{config.provider}'"
            )


class OpenAIChatModelProvider(ChatModelProvider):
    """Chat model provider for OpenAI (ChatOpenAI)."""
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "openai"
    
    def create_chat_model(self, config: LLMConfig) -> Any:
        """
        Create a ChatOpenAI instance from the configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            ChatOpenAI instance
        """
        logger.info(
            "Initializing OpenAI chat model provider",
            model=config.model,
            provider="openai"
        )
        
        try:
            from langchain_openai import ChatOpenAI
            logger.debug("Successfully imported ChatOpenAI from langchain_openai")
        except ImportError as e:
            logger.error(
                "Failed to import ChatOpenAI - package not installed",
                error=str(e)
            )
            raise RuntimeError(
                f"Failed to import ChatOpenAI. Install with: uv pip install langchain-openai"
            ) from e
        
        self.validate_config(config)
        logger.debug(
            "Configuration validated successfully",
            provider="openai",
            model=config.model
        )
        
        # Build kwargs for ChatOpenAI
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "streaming": config.streaming,
        }
        
        # Add optional parameters with OpenAI-specific naming
        if config.api_base:
            kwargs["openai_api_base"] = config.api_base
        if config.api_key:
            kwargs["openai_api_key"] = config.api_key
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        
        # Merge provider-specific parameters
        kwargs.update(config.provider_specific_params)
        
        logger.info(
            "Creating ChatOpenAI instance",
            model=config.model,
            has_custom_endpoint=bool(config.api_base),
            temperature=config.temperature
        )
        
        try:
            chat_model = ChatOpenAI(**kwargs)
            logger.info(
                "ChatOpenAI instance created successfully",
                model=config.model
            )
            return chat_model
        except Exception as e:
            logger.error(
                "Failed to create ChatOpenAI instance",
                model=config.model,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def get_required_dependencies(self) -> List[str]:
        """Get required dependencies."""
        return ["langchain-openai"]


class AzureOpenAIChatModelProvider(ChatModelProvider):
    """Chat model provider for Azure OpenAI (AzureChatOpenAI)."""
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "azure_openai"
    
    def create_chat_model(self, config: LLMConfig) -> Any:
        """
        Create an AzureChatOpenAI instance from the configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            AzureChatOpenAI instance
        """
        logger.info(
            "Initializing Azure OpenAI chat model provider",
            model=config.model,
            provider="azure_openai"
        )
        
        try:
            from langchain_openai import AzureChatOpenAI
            logger.debug("Successfully imported AzureChatOpenAI from langchain_openai")
        except ImportError as e:
            logger.error(
                "Failed to import AzureChatOpenAI - package not installed",
                error=str(e)
            )
            raise RuntimeError(
                f"Failed to import AzureChatOpenAI. Install with: uv pip install langchain-openai"
            ) from e
        
        self.validate_config(config)
        
        # Build kwargs for AzureChatOpenAI
        kwargs = {
            "temperature": config.temperature,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "streaming": config.streaming,
        }
        
        # Azure-specific parameters
        # api_base becomes azure_endpoint
        if config.api_base:
            kwargs["azure_endpoint"] = config.api_base
        
        # api_key becomes azure_api_key or openai_api_key
        if config.api_key:
            kwargs["openai_api_key"] = config.api_key
        
        # model can be used as deployment_name if not provided in provider_specific_params
        if "azure_deployment" in config.provider_specific_params:
            kwargs["azure_deployment"] = config.provider_specific_params["azure_deployment"]
        elif "deployment_name" in config.provider_specific_params:
            kwargs["azure_deployment"] = config.provider_specific_params["deployment_name"]
        else:
            # Use model as deployment name if not specified
            kwargs["azure_deployment"] = config.model
        
        # API version is required for Azure
        if "api_version" in config.provider_specific_params:
            kwargs["openai_api_version"] = config.provider_specific_params["api_version"]
        elif "openai_api_version" in config.provider_specific_params:
            kwargs["openai_api_version"] = config.provider_specific_params["openai_api_version"]
        else:
            # Default to a recent stable version
            kwargs["openai_api_version"] = "2024-02-01"
        
        # Optional model name (can be different from deployment)
        if "model_name" in config.provider_specific_params:
            kwargs["model_name"] = config.provider_specific_params["model_name"]
        
        # Handle max_tokens vs max_completion_tokens based on API version
        # Azure OpenAI API versions >= 2024-08-01 prefer max_completion_tokens
        # Older versions use max_tokens
        if config.max_tokens:
            api_version = kwargs.get("openai_api_version", "")
            # For newer API versions, use max_completion_tokens
            if api_version >= "2024-08-01":
                kwargs["max_completion_tokens"] = config.max_tokens
                logger.debug(
                    "Using max_completion_tokens for Azure OpenAI",
                    api_version=api_version,
                    max_completion_tokens=config.max_tokens
                )
            else:
                kwargs["max_tokens"] = config.max_tokens
                logger.debug(
                    "Using max_tokens for Azure OpenAI",
                    api_version=api_version,
                    max_tokens=config.max_tokens
                )
        
        # Merge any other provider-specific parameters
        # Skip the ones we've already handled
        handled_params = {"azure_deployment", "deployment_name", "api_version", "openai_api_version", "model_name"}
        for key, value in config.provider_specific_params.items():
            if key not in handled_params:
                kwargs[key] = value
        
        logger.info(
            "Creating AzureChatOpenAI instance",
            deployment=kwargs.get("azure_deployment"),
            api_version=kwargs.get("openai_api_version")
        )
        
        try:
            chat_model = AzureChatOpenAI(**kwargs)
            logger.info(
                "AzureChatOpenAI instance created successfully",
                deployment=kwargs.get("azure_deployment")
            )
            return chat_model
        except Exception as e:
            logger.error(
                "Failed to create AzureChatOpenAI instance",
                deployment=kwargs.get("azure_deployment"),
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def get_required_dependencies(self) -> List[str]:
        """Get required dependencies."""
        return ["langchain-openai"]


class OllamaChatModelProvider(ChatModelProvider):
    """Chat model provider for Ollama (ChatOllama)."""
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "ollama"
    
    def create_chat_model(self, config: LLMConfig) -> Any:
        """
        Create a ChatOllama instance from the configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            ChatOllama instance
        """
        logger.info(
            "Initializing Ollama chat model provider",
            model=config.model,
            provider="ollama"
        )
        
        try:
            from langchain_ollama import ChatOllama
            logger.debug("Successfully imported ChatOllama from langchain_ollama")
        except ImportError as e:
            logger.error(
                "Failed to import ChatOllama - package not installed",
                error=str(e)
            )
            raise RuntimeError(
                f"Failed to import ChatOllama. Install with: uv pip install langchain-ollama"
            ) from e
        
        self.validate_config(config)
        
        # Build kwargs for ChatOllama
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
            "timeout": config.timeout,
        }
        
        # Ollama-specific parameters
        if config.api_base:
            kwargs["base_url"] = config.api_base
        
        # num_predict is Ollama's equivalent to max_tokens
        if config.max_tokens:
            kwargs["num_predict"] = config.max_tokens
        
        # Merge provider-specific parameters
        kwargs.update(config.provider_specific_params)
        
        logger.info(
            "Creating ChatOllama instance",
            model=config.model,
            base_url=config.api_base or "http://localhost:11434"
        )
        
        try:
            chat_model = ChatOllama(**kwargs)
            logger.info(
                "ChatOllama instance created successfully",
                model=config.model
            )
            return chat_model
        except Exception as e:
            logger.error(
                "Failed to create ChatOllama instance",
                model=config.model,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def get_required_dependencies(self) -> List[str]:
        """Get required dependencies."""
        return ["langchain-ollama"]


class LLMService:
    """Service for managing and invoking Large Language Models with multiple providers."""
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_dir: str = "./cache/llm_responses",
        cache_size_mb: int = 1024,
        cache_ttl_seconds: int = 3600,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        default_timeout_seconds: float = 60.0
    ):
        self._providers: Dict[str, ChatModelProvider] = {}
        self._llms: Dict[str, Any] = {}
        self._configs: Dict[str, LLMConfig] = {}
        self._initialized = False
        
        # Caching configuration
        self._enable_caching = enable_caching
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: Optional[Cache] = None
        if enable_caching:
            self._cache = Cache(
                directory=cache_dir,
                size_limit=cache_size_mb * 1024 * 1024
            )
        
        # Retry configuration
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._default_timeout_seconds = default_timeout_seconds
        
        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._total_response_time = 0.0
        
    async def initialize(self, auto_register_providers: bool = True) -> None:
        """
        Initialize the LLM service and optionally register default providers.
        
        Args:
            auto_register_providers: If True, register default providers (OpenAI, Azure, Ollama)
        """
        if self._initialized:
            return
        
        logger.info("Initializing LLM service")
        
        # Register default providers if requested
        if auto_register_providers:
            self.register_provider(OpenAIChatModelProvider())
            self.register_provider(AzureOpenAIChatModelProvider())
            self.register_provider(OllamaChatModelProvider())
        
        self._initialized = True
        logger.info(
            "LLM service initialized",
            providers=list(self._providers.keys()),
            registered_llms=len(self._llms)
        )
        
    async def shutdown(self) -> None:
        """Shutdown the LLM service and cleanup resources."""
        logger.info("Shutting down LLM service")
        
        if self._cache:
            self._cache.close()
        
        self._providers.clear()
        self._llms.clear()
        self._configs.clear()
        self._initialized = False
        
    def register_provider(self, provider: ChatModelProvider) -> None:
        """
        Register a chat model provider.
        
        Args:
            provider: Chat model provider instance
        """
        provider_name = provider.get_provider_name()
        self._providers[provider_name] = provider
        
        logger.info(
            "Chat model provider registered",
            provider=provider_name,
            dependencies=provider.get_required_dependencies()
        )
    
    def unregister_provider(self, provider_name: str) -> bool:
        """
        Unregister a chat model provider.
        
        Args:
            provider_name: Name of the provider to unregister
            
        Returns:
            bool: True if provider was unregistered
        """
        if provider_name in self._providers:
            # Remove any LLMs using this provider
            llms_to_remove = [
                name for name, config in self._configs.items()
                if config.provider == provider_name
            ]
            for llm_name in llms_to_remove:
                self.unregister_llm(llm_name)
            
            del self._providers[provider_name]
            logger.info("Chat model provider unregistered", provider=provider_name)
            return True
        return False
    
    def list_providers(self) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def get_provider(self, provider_name: str) -> Optional[ChatModelProvider]:
        """
        Get a provider by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            ChatModelProvider instance or None if not found
        """
        return self._providers.get(provider_name)
        
    def register_llm(self, name: str, llm_config: LLMConfig) -> None:
        """
        Register an LLM configuration by name.
        
        Args:
            name: Unique name for the LLM instance
            llm_config: Configuration for the LLM
            
        Raises:
            ValueError: If provider not found or configuration invalid
        """
        logger.info(
            "Starting LLM registration",
            name=name,
            provider=llm_config.provider,
            model=llm_config.model
        )
        
        # Get the provider
        provider = self._providers.get(llm_config.provider)
        if not provider:
            available_providers = list(self._providers.keys())
            error_msg = (
                f"Provider '{llm_config.provider}' not found. "
                f"Available providers: {available_providers}"
            )
            logger.error(
                "Provider not found during LLM registration",
                name=name,
                requested_provider=llm_config.provider,
                available_providers=available_providers
            )
            raise ValueError(error_msg)
        
        try:
            # Validate configuration
            provider.validate_config(llm_config)
            logger.debug(
                "Configuration validation passed",
                name=name,
                provider=llm_config.provider
            )
            
            # Create the LLM instance
            llm = provider.create_chat_model(llm_config)
            
            # Store the instance and config
            self._llms[name] = llm
            self._configs[name] = llm_config
            
            logger.info(
                "LLM registered successfully",
                name=name,
                provider=llm_config.provider,
                model=llm_config.model,
                total_registered_llms=len(self._llms)
            )
            
        except Exception as e:
            logger.error(
                "Failed to register LLM",
                name=name,
                provider=llm_config.provider,
                model=llm_config.model,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
            
    def unregister_llm(self, name: str) -> bool:
        """
        Unregister an LLM by name.
        
        Args:
            name: Name of the LLM to unregister
            
        Returns:
            bool: True if LLM was unregistered
        """
        if name in self._llms:
            provider = self._configs[name].provider
            del self._llms[name]
            del self._configs[name]
            logger.info("LLM unregistered", name=name, provider=provider)
            return True
        return False
        
    def get_llm(self, name: str) -> Optional[Any]:
        """
        Get an LLM instance by name.
        
        Args:
            name: Name of the LLM
            
        Returns:
            LLM instance or None if not found
        """
        return self._llms.get(name)
        
    def list_llms(self) -> List[str]:
        """
        List all registered LLM names.
        
        Returns:
            List of LLM names
        """
        return list(self._llms.keys())
        
    def get_llm_config(self, name: str) -> Optional[LLMConfig]:
        """
        Get the configuration for an LLM.
        
        Args:
            name: Name of the LLM
            
        Returns:
            LLM configuration or None if not found
        """
        return self._configs.get(name)
    
    def _generate_cache_key(
        self,
        name: str,
        messages: List[Dict[str, str]],
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for request."""
        config = self._configs[name]
        cache_data = {
            "name": name,
            "provider": config.provider,
            "model": config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired."""
        if not self._cache:
            return None
        
        try:
            cached_data = self._cache.get(cache_key)
            if cached_data:
                if cached_data.get("expires_at", 0) > time.time():
                    response_data = cached_data["response"]
                    response_data["cached"] = True
                    return LLMResponse(**response_data)
                else:
                    self._cache.delete(cache_key)
        except Exception as e:
            logger.warning("Cache retrieval error", error=str(e))
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """Cache the response."""
        if not self._cache:
            return
        
        try:
            cache_data = {
                "response": response.model_dump(exclude={"cached"}),
                "expires_at": time.time() + self._cache_ttl_seconds
            }
            self._cache.set(cache_key, cache_data)
        except Exception as e:
            logger.warning("Cache storage error", error=str(e))
    
    def _classify_error(self, error: Exception) -> LLMServiceError:
        """Classify an exception into an LLMServiceError."""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Timeout errors
        if isinstance(error, asyncio.TimeoutError):
            return LLMServiceError(
                f"Request timed out: {error_str}",
                error_code="timeout",
                retryable=True
            )
        
        # HTTP errors (from httpx or requests)
        try:
            import httpx
            if isinstance(error, httpx.HTTPStatusError):
                status_code = error.response.status_code
                if status_code >= 500:
                    return LLMServiceError(
                        f"Server error: {status_code}",
                        error_code="server_error",
                        retryable=True
                    )
                elif status_code == 429:
                    return LLMServiceError(
                        "Rate limit exceeded",
                        error_code="rate_limit",
                        retryable=True
                    )
                else:
                    return LLMServiceError(
                        f"HTTP error: {status_code}",
                        error_code="http_error",
                        retryable=False
                    )
            elif isinstance(error, httpx.RequestError):
                return LLMServiceError(
                    f"Network error: {error_str}",
                    error_code="network_error",
                    retryable=True
                )
        except ImportError:
            pass
        
        # Check for common error patterns in message
        if "timeout" in error_str.lower():
            return LLMServiceError(
                f"Timeout: {error_str}",
                error_code="timeout",
                retryable=True
            )
        elif "rate limit" in error_str.lower():
            return LLMServiceError(
                f"Rate limit: {error_str}",
                error_code="rate_limit",
                retryable=True
            )
        elif "connection" in error_str.lower() or "network" in error_str.lower():
            return LLMServiceError(
                f"Network error: {error_str}",
                error_code="network_error",
                retryable=True
            )
        
        # Default to non-retryable unknown error
        return LLMServiceError(
            f"Unexpected error ({error_type}): {error_str}",
            error_code="unknown",
            retryable=False
        )
    
    async def _invoke_with_retry(
        self,
        llm: Any,
        messages: List[Any],
        config: LLMConfig,
        timeout: float,
        **kwargs
    ) -> Any:
        """Invoke LLM with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    llm.ainvoke(messages, **kwargs),
                    timeout=timeout
                )
                return response
                
            except Exception as e:
                classified_error = self._classify_error(e)
                last_exception = classified_error
                
                if not classified_error.retryable or attempt == self._max_retries:
                    break
                
                delay = self._retry_delay_seconds * (2 ** attempt)
                
                logger.warning(
                    "LLM request failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=self._max_retries + 1,
                    error=str(classified_error),
                    error_code=classified_error.error_code,
                    retry_delay=delay
                )
                
                await asyncio.sleep(delay)
        
        self._error_count += 1
        raise last_exception or LLMServiceError(
            "All retry attempts failed",
            error_code="max_retries_exceeded"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._request_count),
            "average_response_time": self._total_response_time / max(1, self._request_count - self._cache_hits),
            "error_rate": self._error_count / max(1, self._request_count)
        }
    
    async def health_check(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform health check of an LLM or all LLMs.
        
        Args:
            name: Optional LLM name to check. If None, checks first available LLM.
            
        Returns:
            Health check results
        """
        if name is None:
            llm_names = self.list_llms()
            if not llm_names:
                return {
                    "status": "unhealthy",
                    "error": "No LLMs registered"
                }
            name = llm_names[0]
        
        llm = self.get_llm(name)
        if not llm:
            return {
                "status": "unhealthy",
                "llm_name": name,
                "error": f"LLM '{name}' not found"
            }
        
        config = self._configs[name]
        
        try:
            start_time = time.time()
            response = await self.invoke(
                name=name,
                messages="Hello, can you respond with 'OK'?",
                max_tokens=10,
                temperature=0.0
            )
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "llm_name": name,
                "provider": config.provider,
                "model": config.model,
                "response_time": response_time,
                "response_length": len(response.content),
                "cached": response.cached
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "llm_name": name,
                "provider": config.provider,
                "model": config.model,
                "error": str(e),
                "error_type": type(e).__name__
            }
        
    async def invoke(
        self,
        name: str,
        messages: Union[List[Dict[str, str]], str],
        timeout: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Invoke an LLM with a prompt.
        
        Args:
            name: Name of the LLM to invoke
            messages: Either a list of message dicts (OpenAI format) or a single string prompt
            timeout: Optional timeout in seconds (uses default if not specified)
            **kwargs: Additional parameters for the invocation
            
        Returns:
            LLMResponse with the generated content
            
        Raises:
            ValueError: If LLM not found
            LLMServiceError: If invocation fails
        """
        self._request_count += 1
        start_time = time.time()
        
        logger.debug(
            "LLM invoke method called",
            name=name,
            message_type=type(messages).__name__
        )
        
        llm = self.get_llm(name)
        if not llm:
            error_msg = f"LLM '{name}' not found. Available LLMs: {self.list_llms()}"
            logger.error(
                "LLM not found",
                name=name,
                available_llms=self.list_llms()
            )
            raise ValueError(error_msg)
        
        config = self._configs[name]
        
        # Convert string prompt to message format if needed
        if isinstance(messages, str):
            logger.debug(
                "Converting string prompt to message format",
                name=name,
                prompt_length=len(messages)
            )
            messages = [{"role": "user", "content": messages}]
        
        # Generate cache key
        cache_key = self._generate_cache_key(name, messages, kwargs)
        
        # Check cache
        if self._enable_caching:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self._cache_hits += 1
                logger.debug("Cache hit for request", name=name)
                return cached_response
        
        try:
            # Convert to LangChain message format
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:  # user or default
                    langchain_messages.append(HumanMessage(content=content))
            
            logger.debug(
                "Invoking LLM",
                name=name,
                provider=config.provider,
                model=config.model,
                message_count=len(langchain_messages)
            )
            
            # Handle Azure OpenAI max_tokens vs max_completion_tokens
            if config.provider == "azure_openai" and "max_tokens" in kwargs:
                api_version = config.provider_specific_params.get("api_version", "")
                if api_version >= "2024-08-01":
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    logger.debug(
                        "Converted max_tokens to max_completion_tokens for Azure OpenAI",
                        api_version=api_version,
                        max_completion_tokens=kwargs["max_completion_tokens"]
                    )
            
            # Use default timeout if not specified
            if timeout is None:
                timeout = self._default_timeout_seconds
            
            # Invoke with retry logic
            response = await self._invoke_with_retry(
                llm=llm,
                messages=langchain_messages,
                config=config,
                timeout=timeout,
                **kwargs
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            self._total_response_time += response_time
            
            # Extract token usage if available
            usage = None
            if hasattr(response, "response_metadata") and response.response_metadata:
                token_usage = response.response_metadata.get("token_usage")
                if token_usage:
                    usage = {
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0),
                    }
            
            # Build response
            llm_response = LLMResponse(
                content=response.content,
                model=config.model,
                provider=config.provider,
                finish_reason=getattr(response, "finish_reason", None),
                usage=usage,
                response_time=response_time,
                cached=False,
                metadata={
                    "llm_name": name,
                }
            )
            
            # Cache the response
            if self._enable_caching:
                self._cache_response(cache_key, llm_response)
            
            logger.info(
                "LLM invocation completed",
                name=name,
                provider=config.provider,
                model=config.model,
                content_length=len(response.content),
                response_time=response_time,
                cached=False,
                usage=usage
            )
            
            return llm_response
            
        except LLMServiceError:
            raise
        except Exception as e:
            self._error_count += 1
            classified_error = self._classify_error(e)
            logger.error(
                "LLM invocation failed",
                name=name,
                provider=config.provider,
                model=config.model,
                error=str(classified_error),
                error_code=classified_error.error_code,
                error_type=type(e).__name__
            )
            raise classified_error
    
    async def _stream_with_retry(
        self,
        llm: Any,
        messages: List[Any],
        config: LLMConfig,
        timeout: float,
        **kwargs
    ):
        """Stream from LLM with retry logic."""
        last_exception = None
        
        for attempt in range(self._max_retries + 1):
            try:
                async def stream_with_timeout():
                    async for chunk in llm.astream(messages, **kwargs):
                        yield chunk
                
                async for chunk in stream_with_timeout():
                    yield chunk
                return
                
            except Exception as e:
                classified_error = self._classify_error(e)
                last_exception = classified_error
                
                if not classified_error.retryable or attempt == self._max_retries:
                    break
                
                delay = self._retry_delay_seconds * (2 ** attempt)
                
                logger.warning(
                    "LLM stream failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=self._max_retries + 1,
                    error=str(classified_error),
                    error_code=classified_error.error_code,
                    retry_delay=delay
                )
                
                await asyncio.sleep(delay)
        
        self._error_count += 1
        raise last_exception or LLMServiceError(
            "All retry attempts failed",
            error_code="max_retries_exceeded"
        )
    
    async def stream(
        self,
        name: str,
        messages: Union[List[Dict[str, str]], str],
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        Stream responses from an LLM.
        
        Args:
            name: Name of the LLM to invoke
            messages: Either a list of message dicts (OpenAI format) or a single string prompt
            timeout: Optional timeout in seconds (uses default if not specified)
            **kwargs: Additional parameters for the invocation
            
        Yields:
            Chunks of generated content
            
        Raises:
            ValueError: If LLM not found
            LLMServiceError: If streaming fails
        """
        self._request_count += 1
        start_time = time.time()
        
        llm = self.get_llm(name)
        if not llm:
            raise ValueError(f"LLM '{name}' not found. Available LLMs: {self.list_llms()}")
        
        config = self._configs[name]
        
        try:
            # Convert string prompt to message format if needed
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # Convert to LangChain message format
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:  # user or default
                    langchain_messages.append(HumanMessage(content=content))
            
            logger.debug(
                "Streaming from LLM",
                name=name,
                provider=config.provider,
                model=config.model,
                message_count=len(langchain_messages)
            )
            
            # Handle Azure OpenAI max_tokens vs max_completion_tokens
            if config.provider == "azure_openai" and "max_tokens" in kwargs:
                api_version = config.provider_specific_params.get("api_version", "")
                if api_version >= "2024-08-01":
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    logger.debug(
                        "Converted max_tokens to max_completion_tokens for Azure OpenAI streaming",
                        api_version=api_version,
                        max_completion_tokens=kwargs["max_completion_tokens"]
                    )
            
            # Use default timeout if not specified
            if timeout is None:
                timeout = self._default_timeout_seconds
            
            # Stream with retry logic
            async for chunk in self._stream_with_retry(
                llm=llm,
                messages=langchain_messages,
                config=config,
                timeout=timeout,
                **kwargs
            ):
                yield chunk
            
            # Track response time
            response_time = time.time() - start_time
            self._total_response_time += response_time
            
            logger.info(
                "LLM streaming completed",
                name=name,
                provider=config.provider,
                model=config.model,
                response_time=response_time
            )
            
        except LLMServiceError:
            raise
        except Exception as e:
            self._error_count += 1
            classified_error = self._classify_error(e)
            logger.error(
                "LLM streaming failed",
                name=name,
                provider=config.provider,
                model=config.model,
                error=str(classified_error),
                error_code=classified_error.error_code,
                error_type=type(e).__name__
            )
            raise classified_error


# Global singleton instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance.
    
    Returns:
        LLMService instance
    """
    global _llm_service
    
    if _llm_service is None:
        _llm_service = LLMService()
        await _llm_service.initialize()
    
    return _llm_service


async def shutdown_llm_service() -> None:
    """Shutdown the global LLM service instance."""
    global _llm_service
    
    if _llm_service:
        await _llm_service.shutdown()
        _llm_service = None
        logger.info("LLM service shutdown completed")

