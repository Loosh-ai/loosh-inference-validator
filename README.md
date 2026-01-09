# Loosh Inference Validator

A Bittensor subnet validator for LLM inference evaluation. This validator evaluates miner responses and allocates emissions based on response quality.

## Project Structure

```
loosh-inference-validator/
├── validator/               # Validator code
│   ├── challenge/         # Challenge handling
│   ├── challenge_api/     # Challenge API integration
│   ├── db/               # Database operations
│   ├── evaluation/       # Response evaluation
│   ├── endpoints/        # API endpoints
│   ├── miner_api/       # Miner API client
│   ├── network/         # Bittensor network code
│   ├── scripts/         # Utility scripts
│   ├── utils/           # Utility functions
│   ├── config/           # Configuration
│   │   ├── config.py    # Validator configuration
│   │   └── shared_config.py # Shared config utilities
│   └── main.py          # Validator entry point
├── docker/               # Docker configuration
│   └── Dockerfile        # Dockerfile
└── pyproject.toml        # Project configuration
```

## Features

- Challenge generation and distribution
- Response evaluation using consensus and similarity metrics
- Heatmap visualization of response similarity
- Emissions allocation based on response quality and speed
- Database storage of challenges, responses, and evaluation results
- Integration with Challenge API

## Requirements

- Python 3.12+
- uv (Python package installer) - [Installation instructions](https://github.com/astral-sh/uv)
- uvicorn (ASGI web server) - Included in dependencies, used to run the FastAPI application
- fiber (Bittensor network library) - Required for Bittensor network operations (installed manually due to dependency conflicts)
- Bittensor wallet with sufficient stake
- Access to Challenge API
- OpenAI-compatible inference endpoint - Required for generating consensus narratives during evaluation. Can be:
  - OpenAI API (default)
  - Azure OpenAI
  - Ollama (local or remote)
  - Any OpenAI-compatible API endpoint

## Installation

1. Clone the repository:
```bash
git clone https://github.com/loosh-ai/loosh-inference-validator.git
cd loosh-inference-validator
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

3. Install dependencies:
```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies from `pyproject.toml`.

4. Install fiber (required dependency):
```bash
# Activate the virtual environment first
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install fiber from git repository
uv pip install "git+https://github.com/rayonlabs/fiber.git@production#egg=fiber[chain]"
```

Or using regular pip:
```bash
pip install "git+https://github.com/rayonlabs/fiber.git@production#egg=fiber[chain]"
```

**Note:** Fiber must be installed manually due to a dependency version conflict with bittensor:


To activate the virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

## Configuration

Create a `.env` file in the project root by copying the example file:

```bash
cp env.example .env
```

Then edit `.env` and update the values according to your setup. See `env.example` for all available configuration options with descriptions.

**Note:** Fiber only supports wallets in `~/.bittensor/wallets`. Custom wallet paths are not supported.

## Running

### Starting the Validator

The validator runs as a FastAPI application using uvicorn. The API server exposes endpoints for challenge submission and validator management, while the main validation loop runs in the background.

#### Direct Execution

```bash
PYTHONPATH=. uv run uvicorn validator.validator_server:app --host 0.0.0.0 --port 8000
```

Or if you have activated the virtual environment:

```bash
source .venv/bin/activate  # Linux/Mac
uvicorn validator.validator_server:app --host 0.0.0.0 --port 8000
```

The host and port can be configured via environment variables (`API_HOST` and `API_PORT` in your `.env` file). Alternatively, you can run it directly:

```bash
PYTHONPATH=. uv run python -m validator.validator_server
```

This will automatically read `API_HOST` and `API_PORT` from your `.env` file.

#### Using PM2 (Recommended for Production)

PM2 is a process manager that provides automatic restarts, logging, and monitoring. It's recommended for production deployments.

**Prerequisites:**
```bash
npm install -g pm2
```

**Starting with PM2:**

1. Navigate to the project directory:
```bash
cd loosh-inference-validator
```

2. Start the validator:
```bash
pm2 start PM2/ecosystem.config.js
```

**Note:** The PM2 configuration runs the validator using uvicorn. Make sure to update `PM2/ecosystem.config.js` to use uvicorn if it's not already configured.

3. Check status:
```bash
pm2 status
```

4. View logs:
```bash
pm2 logs loosh-inference-validator
```

5. Stop the validator:
```bash
pm2 stop loosh-inference-validator
```

6. Restart the validator:
```bash
pm2 restart loosh-inference-validator
```

7. Delete from PM2:
```bash
pm2 delete loosh-inference-validator
```

**PM2 Configuration:**

The PM2 configuration file is located at `PM2/ecosystem.config.js`. It runs the validator using uvicorn. You can customize it by:

- Setting `PYTHON_INTERPRETER` environment variable to use a specific Python interpreter (e.g., `.venv/bin/python3`)
- Setting `VALIDATOR_WORKDIR` environment variable to specify the working directory
- Setting `API_HOST` and `API_PORT` environment variables to configure the server address
- Adjusting memory limits, restart policies, and logging paths in the config file

**Example with virtual environment:**
```bash
PYTHON_INTERPRETER=./.venv/bin/python3 pm2 start PM2/ecosystem.config.js
```

**Example with custom port:**
```bash
API_PORT=8020 pm2 start PM2/ecosystem.config.js
```

**Note:** The logs directory (`./logs/`) will be created automatically by PM2 if it doesn't exist. Logs are stored in:
- `logs/validator-error.log` - Error logs
- `logs/validator-out.log` - Standard output logs
- `logs/validator-combined.log` - Combined logs with timestamps

**PM2 Useful Commands:**
- `pm2 monit` - Real-time monitoring dashboard
- `pm2 save` - Save current process list
- `pm2 startup` - Generate startup script for system boot
- `pm2 logs` - View all logs
- `pm2 flush` - Clear all logs

## Docker Deployment

### Building the Docker Image

```bash
cd docker
docker build -t loosh-inference-validator .
```

### Running with Docker

```bash
docker run -d \
  --name loosh-validator \
  -p 8000:8000 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets \
  -v $(pwd)/validator.db:/app/validator.db \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8000 \
  loosh-inference-validator
```

**Note:** The Docker container runs the validator using uvicorn via `validator.validator_server:app`. The API host and port can be configured via `API_HOST` and `API_PORT` environment variables.

## API Endpoints

- `GET /availability` - Check validator availability
- `POST /challenges` - Submit a challenge to the validator (push mode)
- `GET /challenges/stats` - Get challenge queue statistics
- `POST /register` - Register with validator
- `GET /stats` - Get validator statistics
- `POST /set_running` - Set running status

## Inference Configuration

The validator uses LLM inference for generating consensus narratives during the evaluation process. The validator provides flexible options for configuring inference endpoints and models.

### LLM Service

The validator uses `llm_service.py` (`validator/evaluation/Recording/llm_service.py`) to manage LLM inference. This service supports multiple providers and allows you to configure different endpoints for inference:

- **OpenAI**: Standard OpenAI API endpoints (including custom endpoints)
- **Azure OpenAI**: Azure-hosted OpenAI models
- **Ollama**: Local or remote Ollama instances

### Configuring Inference Endpoints

The LLM service allows you to configure custom API endpoints through the `LLMConfig` class:

```python
from validator.evaluation.Recording.llm_service import LLMService, LLMConfig

# Initialize the service
llm_service = LLMService()
await llm_service.initialize()

# Register an LLM with a custom endpoint
llm_service.register_llm(
    name="my-openai",
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key",
        api_base="https://your-custom-endpoint.com/v1",  # Custom endpoint
        temperature=0.7,
        max_tokens=800
    )
)
```

### Supported Providers

- **OpenAI** (`provider="openai"`): Uses `langchain-openai` package
  - Supports custom `api_base` for self-hosted or proxy endpoints
  - Requires `langchain-openai` package

- **Azure OpenAI** (`provider="azure_openai"`): Uses `langchain-openai` package
  - Configure via `api_base` (Azure endpoint) and `provider_specific_params`
  - Requires `langchain-openai` package

- **Ollama** (`provider="ollama"`): Uses `langchain-ollama` package
  - Supports local (`http://localhost:11434`) or remote Ollama instances
  - Requires `langchain-ollama` package

### Local Inference Setup

For running inference locally, see `min_compute.yml` which describes:

- **Hardware Requirements**: Minimum and recommended CPU, GPU, memory, and storage specifications
- **Recommended Models**: 
  - **Primary**: Qwen/Qwen2.5-14B-Instruct (recommended for H100 80GB)
  - **Minimum**: Qwen/Qwen2.5-7B-Instruct-AWQ (for A10 24GB)
- **Deployment Notes**: Configuration recommendations for vLLM/TGI with continuous batching

The `min_compute.yml` file provides detailed specifications for:
- GPU requirements (VRAM, compute capability)
- Model recommendations with quantization options
- Performance characteristics
- Deployment best practices

### Environment Variables

Inference configuration can be set via environment variables in your `.env` file:

```env
# OpenAI Configuration (for evaluation)
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4
```

For custom endpoints, you can configure the LLM service programmatically as shown above.

## Challenge Mode

The validator operates in **push mode** only. Challenges are pushed to the validator via the `POST /challenges` endpoint, where they are queued for processing. The validator does not pull challenges from the Challenge API.

### Submitting Challenges

Challenges can be submitted to the validator using the `/challenges` endpoint:

```bash
curl -X POST http://localhost:8000/challenges \
  -H "Content-Type: application/json" \
  -d '{
    "id": "challenge-123",
    "prompt": "Explain the concept of machine learning in simple terms.",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 512
  }'
```

See the [API Endpoints](#api-endpoints) section for more details.

## Evaluation Process

For a detailed explanation of the evaluation pipeline, see [EVALUATION_PROCESS.md](EVALUATION_PROCESS.md).

High-level overview:

1. Challenges are pushed to the validator via `POST /challenges` endpoint (push mode)
2. Validator processes challenges from its internal queue
3. Challenges are sent to selected miners
4. Miners generate responses using their LLM backends
5. Validator collects responses and calculates embeddings
6. Consensus score and heatmap are generated using LLM inference (via `llm_service.py`)
7. Emissions are allocated based on response quality and speed
8. Results are stored in the database and sent to Challenge API

The evaluation process includes:
- **Embedding Generation**: Responses are converted to embeddings for similarity analysis
- **Quality Filtering**: Outlier detection, length filtering, and clustering to identify consensus
- **Consensus Measurement**: Pairwise similarity analysis to determine agreement among responses
- **Individual Scoring**: Composite scores based on consensus alignment, quality, and confidence
- **Heatmap Visualization**: Similarity matrix visualization of response relationships
- **Narrative Generation**: LLM-generated summaries of the evaluation process
- **Emissions Calculation**: Reward distribution based on speed, consensus participation, and quality

## Database

The validator uses SQLite by default to store:
- Miner information
- Challenges and responses
- Evaluation results
- Statistics

The database files are **automatically created** when the validator starts if they don't exist. The database file locations are configured via environment variables:
- `DB_PATH` - Main validator database (default: `validator.db`)
- `USERS_DB_PATH` - Users database (default: `users.db`)

Both databases and their schemas are automatically initialized on first startup.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

