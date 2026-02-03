import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import bittensor as bt

bt.trace()

# Add parent of db_adapter to path (Docker: /app, Local: loosh-stats/)
DB_ADAPTER_DOCKER_PARENT = Path("/app")
DB_ADAPTER_LOCAL_PARENT = Path(__file__).parent.parent.parent / "loosh-stats"

if (DB_ADAPTER_DOCKER_PARENT / "db_adapter").exists():
    DB_ADAPTER_PARENT = DB_ADAPTER_DOCKER_PARENT
else:
    DB_ADAPTER_PARENT = DB_ADAPTER_LOCAL_PARENT

sys.path.insert(0, str(DB_ADAPTER_PARENT))
bt.logging.info(f"DB_ADAPTER_PARENT: {DB_ADAPTER_PARENT}, db_adapter exists: {(DB_ADAPTER_PARENT / 'db_adapter').exists()}")

try:
    from db_adapter.factory import get_adapter
    from db_adapter.adapter import DatabaseAdapter
    bt.logging.info("Successfully imported db_adapter modules")
except ImportError as e:
    bt.logging.error(f"Failed to import db_adapter: {e}")
    bt.logging.error(f"sys.path: {sys.path}")
    # Define stubs to allow the module to load
    DatabaseAdapter = None  # type: ignore
    get_adapter = None  # type: ignore

from validator.network.bittensor_node import BittensorNode
# Pull-based challenge fetching removed - only push mode (Fiber) is supported
from validator.challenge.challenge_types import InferenceResponse
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import InferenceValidator


# CONFIG [

from validator.config import get_validator_config

# Database configuration from environment
DB_BACKEND = os.getenv("DB_BACKEND", "redis")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"mysql+pymysql://{os.getenv('MYSQL_USER', 'loosh')}:{os.getenv('MYSQL_PASSWORD', 'loosh_password')}@{os.getenv('MYSQL_HOST', 'localhost')}:{os.getenv('MYSQL_PORT', '3306')}/{os.getenv('MYSQL_DATABASE', 'loosh_stats')}"
)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "status")


def get_db_adapter() -> "DatabaseAdapter | None":
    """Create database adapter based on configuration."""
    if get_adapter is None:
        bt.logging.error("db_adapter not available - get_adapter is None")
        return None
    
    if DB_BACKEND == "mysql":
        return get_adapter("mysql", database_url=DATABASE_URL)
    elif DB_BACKEND == "redis":
        return get_adapter(
            "redis",
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_db=REDIS_DB,
            redis_password=REDIS_PASSWORD,
            redis_prefix=REDIS_PREFIX,
        )
    else:
        return get_adapter("mysql", database_url=DATABASE_URL)


def load_config(type='validator'):
    # Load configuration
    if type == 'validator':
        config = get_validator_config()
    # Only validator type is supported in this repository
    else:
        raise ValueError(f"Invalid configuration type: {type}")
    
    bt.logging.info(f"Starting validator with configuration:")
    bt.logging.info(f"Network: {config.subtensor_network}")
    bt.logging.info(f"Chain Endpoint: {config.subtensor_address}")
    bt.logging.info(f"Subnet: {config.netuid}")
    bt.logging.info(f"Wallet: {config.wallet_name}")
    bt.logging.info(f"Hotkey: {config.hotkey_name}")
    bt.logging.info(f"Challenge Interval: {config.challenge_interval}")
    bt.logging.info(f"Challenge Timeout: {config.challenge_timeout}")
    bt.logging.info(f"Challenge API URL: {config.challenge_api_url}")
    bt.logging.info(f"Challenge API Key: {config.challenge_api_key}")

    return config

# CONFIG ]

# VALIDATOR [

class Validator:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.node = None
        self.db: "DatabaseAdapter | None" = None
        self.db_manager: DatabaseManager | None = None
        self.inference_validator: InferenceValidator | None = None
    
        # Initialize database
        self._init_db()
        self._init_evaluation()

    def _log_status_json(self, status: str, extra: dict | None = None) -> None:
        """Log status in JSON format and optionally save to database."""
        status_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "validator": {
                "network": self.config.subtensor_network,
                "chain_endpoint": self.config.subtensor_address,
                "netuid": self.config.netuid,
                "wallet": self.config.wallet_name,
                "hotkey": self.config.hotkey_name,
            },
            "config": {
                "challenge_interval_seconds": self.config.challenge_interval_seconds,
                "challenge_timeout_seconds": self.config.challenge_timeout_seconds,
                "challenge_api_url": self.config.challenge_api_url,
                "min_miners": self.config.min_miners,
                "max_miners": self.config.max_miners,
            },
        }
        if extra:
            status_data.update(extra)
        
        bt.logging.info(f"STATUS_JSON: {json.dumps(status_data, indent=2)}")
        
        # Save to database if connected
        if self.db:
            try:
                self.db.create_status(json.dumps(status_data))
            except Exception as e:
                bt.logging.warning(f"Failed to save status to database: {e}")

    def _log_miner_challenge_result_json(
        self,
        node_id: int,
        hotkey: str,
        status: str,
        score: float | None = None,
    ) -> None:
        """Log miner challenge result in JSON format."""
        result_data = {
            "type": "miner_challenge_result",
            "node_id": node_id,
            "hotkey": hotkey,
            "status": status,
            "score": score,
            "last_updated": datetime.now().isoformat(),
        }
        
        bt.logging.info(f"MINER_CHALLENGE_RESULT_JSON: {json.dumps(result_data, indent=2)}")
        
        # Save to database if connected
        if self.db:
            try:
                self.db.create_status(json.dumps(result_data))
            except Exception as e:
                bt.logging.warning(f"Failed to save miner challenge result to database: {e}")
    
    def _init_db(self) -> None:
        """Initialize database adapter."""
        try:
            self.db = get_db_adapter()
            if self.db is None:
                bt.logging.warning("db_adapter not available. Continuing without db.")
                return
            self.db.connect()
            bt.logging.info(f"Connected to database backend: {DB_BACKEND}")
        except Exception as e:
            bt.logging.warning(f"Failed to connect to database: {e}. Continuing without db.")
            self.db = None

    def _init_evaluation(self) -> None:
        """Initialize evaluation components."""
        try:
            db_path = os.getenv("VALIDATOR_DB_PATH", "validator.db")
            self.db_manager = DatabaseManager(db_path)
            self.inference_validator = InferenceValidator(self.db_manager)
            bt.logging.info("Evaluation components initialized")
        except Exception as e:
            bt.logging.warning(f"Failed to initialize evaluation components: {e}. Continuing without evaluation.")
            self.db_manager = None
            self.inference_validator = None
    
    def start(self):
        """Start the validator."""

        self.node = BittensorNode(config=self.config)
        self.node.launch(port=8099)

        bt.logging.info("Validator started")
        
        # Log startup status in JSON format
        self._log_status_json("started", {
            "db_backend": DB_BACKEND,
            "db_connected": self.db is not None,
        })

        self.running = True

    async def process(self):
        """
        Process challenges from queue (push mode only).
        
        Note: This method is deprecated. The validator now uses the main_loop() 
        in validator/main.py which consumes challenges from the queue.
        Challenges are received via Fiber-encrypted POST to /fiber/challenge endpoint.
        """
        bt.logging.warning(
            "validator0.py process() method is deprecated. "
            "Use validator/main.py main_loop() instead which supports Fiber-encrypted challenges."
        )
        bt.logging.info("Validator0 process() exiting - use main_loop() for push mode")
        return

    
    def stop(self):
        """Stop the validator."""
        self._log_status_json("stopped")
        
        # Disconnect from database
        if self.db:
            try:
                self.db.disconnect()
                bt.logging.info("Disconnected from database")
            except Exception as e:
                bt.logging.warning(f"Error disconnecting from database: {e}")
        
        self.running = False
    
# 

async def main_loop():
    """Main validator loop."""
    
    try:
        config = load_config(type='validator')
        validator = Validator(config)
        validator.start()

        await validator.process()

    except KeyboardInterrupt:
        validator.stop()
    except Exception as e:
        bt.logging.error(f"Error in main loop: {str(e)}")
    finally:
        validator.stop()
        bt.logging.info("Validator stopped")
    return 0

def run_main_loop():
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    run_main_loop()
    exit(0)
