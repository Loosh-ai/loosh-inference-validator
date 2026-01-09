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
- Bittensor wallet with sufficient stake
- Access to Challenge API

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

#### Direct Execution

```bash
PYTHONPATH=. uv run python -m validator.main
```

Or if you have activated the virtual environment:

```bash
source .venv/bin/activate  # Linux/Mac
python -m validator.main
```

This will automatically use the virtual environment created by `uv sync` and run the validator. The `-m` flag ensures the `validator` module is found correctly in the Python path.

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

The PM2 configuration file is located at `PM2/ecosystem.config.js`. You can customize it by:

- Setting `PYTHON_INTERPRETER` environment variable to use a specific Python interpreter (e.g., `.venv/bin/python3`)
- Setting `VALIDATOR_WORKDIR` environment variable to specify the working directory
- Adjusting memory limits, restart policies, and logging paths in the config file

**Example with virtual environment:**
```bash
PYTHON_INTERPRETER=./.venv/bin/python3 pm2 start PM2/ecosystem.config.js
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

