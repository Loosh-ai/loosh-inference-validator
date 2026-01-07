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
- Bittensor wallet with sufficient stake
- Access to Challenge API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/loosh-ai/loosh-inference-validator.git
cd loosh-inference-validator
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
NETUID=21
SUBTENSOR_NETWORK=finney
SUBTENSOR_ADDRESS=wss://entrypoint-finney.opentensor.ai:443
WALLET_NAME=validator
HOTKEY_NAME=validator
MIN_MINERS=3
MAX_MINERS=10
MIN_STAKE_THRESHOLD=100
CHALLENGE_INTERVAL_SECONDS=300
CHALLENGE_TIMEOUT_SECONDS=120
EVALUATION_TIMEOUT_SECONDS=300
SCORE_THRESHOLD=0.7
WEIGHTS_INTERVAL_SECONDS=1800
DB_PATH=validator.db
CHALLENGE_API_URL=http://localhost:8080
CHALLENGE_API_KEY=your-api-key
API_HOST=0.0.0.0
API_PORT=8000
HEATMAP_UPLOAD_URL=http://localhost:8080/upload
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_MODEL=gpt-4
LOG_LEVEL=INFO
```

## Running

### Starting the Validator

```bash
python validator/main.py
```

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
  loosh-inference-validator
```

## API Endpoints

- `GET /availability` - Check validator availability
- `GET /challenges` - Get challenge information
- `POST /register` - Register with validator
- `GET /stats` - Get validator statistics
- `POST /set_running` - Set running status

## Evaluation Process

1. Validator fetches challenges from Challenge API
2. Challenges are sent to selected miners
3. Miners generate responses using their LLM backends
4. Validator collects responses and calculates embeddings
5. Consensus score and heatmap are generated
6. Emissions are allocated based on response quality and speed
7. Results are stored in the database and sent to Challenge API

## Database

The validator uses SQLite by default to store:
- Miner information
- Challenges and responses
- Evaluation results
- Statistics

The database file location is configured via `DB_PATH` environment variable.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

