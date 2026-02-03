# BITTENSOR NODE IMPLEMENTATION

import bittensor as bt
import time
import socket
from typing import Any, Dict

import yaml, os

# Import selective fiber utilities
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair

# Import validator config
from validator.config import get_validator_config

from validator.network.axon import LooshSubnetSubtensor

# Import Bittensor Synapse
from validator.network.InferenceSynapse import InferenceSynapse, inference, blacklist, priority

bt.trace()

themecolor = 'red'
#themecolor = 'green'
#themecolor = 'orange'

only_success = True

# CONFIG-TEST [

from test_config import create_bittensor_test_config

def load_config_yaml(config_path: str = "test_config.yaml") -> Dict[str, Any]:
    """Load test configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Test config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# CONFIG-TEST ]
# TASKS [

class PullTask(bt.Synapse):
    prompt: str

class ChallengeTask(bt.Synapse):
    prompt: str


# TASKS ]
# BittensorNode [

class BittensorNode:
    """Bittensor node for LLM inference using native axon/dendrite communication."""
    
    def __init__(self, config=None, log_level="debug"):
        """Initialize the bittensor node with configuration."""
        self.config = config or get_validator_config()
        self._axon = None
        self._dendrite = None
        self.hotkey = None
        self.coldkey = None
        self.log_level = log_level

    def launch(self, port):
        self.stageA(port)
        #self.stage3()

    def load_keys(self):
        """Load hotkey and coldkey using fiber utilities."""
        from pathlib import Path
        
        # Note: Fiber only supports wallets in ~/.bittensor/wallets
        wallet_path = Path.home() / ".bittensor" / "wallets" / self.config.wallet_name
        hotkey_path = wallet_path / "hotkeys" / self.config.hotkey_name
        coldkey_path = wallet_path / "coldkey"
        
        try:
            self.hotkey = load_hotkey_keypair(self.config.wallet_name, self.config.hotkey_name)
            self.coldkey = load_coldkeypub_keypair(self.config.wallet_name)
            bt.logging.info(f"Loaded keys for wallet: {self.config.wallet_name}, hotkey: {self.config.hotkey_name}")
        except FileNotFoundError as e:
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Wallet files not found!\n"
                f"{'='*70}\n"
                f"Wallet: {self.config.wallet_name}\n"
                f"Hotkey: {self.config.hotkey_name}\n"
                f"Note: Fiber only supports wallets in ~/.bittensor/wallets\n"
                f"Expected paths:\n"
                f"  - Hotkey: {hotkey_path}\n"
                f"  - Coldkey: {coldkey_path}\n"
                f"\nTo create the wallet, run:\n"
                f"  btcli wallet new_coldkey \\\n"
                f"    --wallet.name {self.config.wallet_name} \\\n"
                f"    --no-use-password --n_words 24\n"
                f"\n  btcli wallet new_hotkey \\\n"
                f"    --wallet.name {self.config.wallet_name} \\\n"
                f"    --hotkey {self.config.hotkey_name} \\\n"
                f"    --no-use-password --n_words 24\n"
                f"{'='*70}\n"
            )
            bt.logging.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Failed to load wallet keys!\n"
                f"{'='*70}\n"
                f"Wallet: {self.config.wallet_name}\n"
                f"Hotkey: {self.config.hotkey_name}\n"
                f"Error: {str(e)}\n"
                f"\nPlease ensure wallet files exist at:\n"
                f"  - Hotkey: {hotkey_path}\n"
                f"  - Coldkey: {coldkey_path}\n"
                f"{'='*70}\n"
            )
            bt.logging.error(error_msg)
            raise
    
    def setup_axon(self, ip=None, port=8099, external_ip=None, external_port=None):
        """Setup the axon for serving requests."""

        if self.log_level == "debug":
            bt.logging.debug(f"Setting up axon on port {port}")
            bt.logging.debug(f"Hotkey: {self.hotkey}")
            bt.logging.debug(f"Coldkey: {self.coldkey}") 
            bt.logging.debug(f"Wallet: {self.config.wallet_name}")
            bt.logging.debug(f"Hotkey name: {self.config.hotkey_name}")

        wallet = bt.wallet(
            name=self.config.wallet.name, 
            hotkey=self.config.wallet.hotkey,
        )

        external_ip = socket.gethostbyname(socket.gethostname())

        self._axon = bt.axon(ip=ip, port=port, external_ip=external_ip, external_port=external_port, wallet=wallet)
#        self._axon = bt.axon(ip=ip, port=port, external_ip=external_ip, external_port=external_port)
        
        # API [

        # Attach the forward, blacklist, and priority functions to the Axon.
        # forward_fn: The function to handle forwarding logic.
        # blacklist_fn: The function to determine if a request should be blacklisted.
        # priority_fn: The function to determine the priority of the request.
        self._axon.attach(
            forward_fn=inference,
            blacklist_fn=blacklist,
            priority_fn=priority
        )

        # API ]

        #bt.logging.info(f"Axon setup complete on port {port}")
    
    def start_axon(self):
        """Start the axon to begin listening for requests."""
        if self._axon is None:
            raise RuntimeError("Axon not setup. Call setup_axon() first.")
        
        self._axon.start()

#        bt.logging.info(f"[green]Axon: {self.axon}[/green]")

        bt.logging.info(f"[{themecolor}]Axon: {self.axon}[/{themecolor}]")
    
    
    def stop_axon(self):
        """Stop the axon."""
        if self._axon is not None:
            self._axon.stop()
            bt.logging.info("Axon stopped")

    def start(self):
        """Start the axon."""
        self.start_axon()

    def stop(self):
        """Stop the axon."""
        self.stop_axon()

    # ACCESSLIST [

    blacklist_hotkeys: set
    blacklist_coldkeys: set
    whitelist_hotkeys: set
    whitelist_coldkeys: set

    def init_accesslist(self):
        self.blacklist_coldkeys = set()
        self.blacklist_hotkeys = set()
        self.whitelist_hotkeys = set()
        self.whitelist_hotkeys = set()
    
        # TODO: load access keys

    def is_blacklisted(self, neuron: bt.NeuronInfoLite):
        coldkey = neuron.coldkey
        hotkey = neuron.hotkey

        return False

        # Blacklist coldkeys that are blacklisted by user
        if coldkey in self.blacklist_coldkeys:
            bt.logging.trace(f"Blacklisted recognized coldkey {coldkey} - with hotkey: {hotkey}")
            return True

        # Blacklist coldkeys that are blacklisted by user or by set of hotkeys
        if hotkey in self.blacklist_hotkeys:
            bt.logging.trace(f"Blacklisted recognized hotkey {hotkey}")
            # Add the coldkey attached to this hotkey in the blacklisted coldkeys
            self.blacklist_hotkeys.add(coldkey)
            return True

        return False
    
    # ACCESSLIST ]
    # METAGRAPH [

    def metagraph(self):
        return self._metagraph

    def updateMetagraph(self):
        # The metagraph holds the state of the network, letting us know about other miners.
        self._metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph()}")
        self.uids = self.metagraph().uids.tolist()
        bt.logging.info(f"Metagraph UIDS: {self.uids}")
        bt.logging.info(f"[green]Metagraph Axons: {self.metagraph().axons}[/green]")

    def metagraph_get_neuron_uid(self, hotkey: str) -> int | None:
        for neuron in self.metagraph().neurons:
            if neuron.hotkey == hotkey:
                return int(neuron.uid)

        return None

    # METAGRAPH ]
    # METAGRAPH select_axons [

    def select_axons(self) -> []:
        self.updateMetagraph()
        valid_queryable = []
        for uid in self.uids:
            neuron: bt.NeuronInfoLite = self.metagraph().neurons[uid]
            axon = self.metagraph().axons[uid]

            bt.logging.info(f"Axon IP: {neuron.axon_info.ip}:{neuron.axon_info.port} {neuron.hotkey}")

#            if neuron.axon_info.ip != "0.0.0.0" and not self.is_blacklisted(neuron=neuron):
            if neuron.axon_info.port != 0 and not self.is_blacklisted(neuron=neuron):
                valid_queryable.append((uid, axon))

        return valid_queryable

    # METAGRAPH get_valid_queryable ]
    # CHECK REGISTRED IN BT [

    def check_miner_registered(self, *, synapse: PullTask) -> int | None:
        miner_uid = self.metagraph_neuron_uid(synapse.dendrite.hotkey)
        if miner_uid is None:
            synapse.cooldown_until = int(time.time()) + 3600
            bt.logging.debug(f"Miner {synapse.dendrite.hotkey} ({synapse.dendrite.ip}) is not registered")
        return miner_uid

    # CHECK REGISTRED IN BT ]
    # PULL TASK [

    async def pull_task(validator_uid: int, wallet: bt.wallet, metagraph: bt.metagraph) -> PullTask:
        async with bt.dendrite(wallet=wallet) as dendrite:
            synapse = PullTask()
            return await dendrite.call(
                target_axon=metagraph.axons[validator_uid], synapse=synapse, deserialize=False, timeout=12.0
            )

    # PULL TASK ]
    # SEND CHALLENGE TASK [

    challenge_timeout = 12.0

    @property
    def dendrite(self):
        return self._dendrite

    @property
    def axon(self):
        return self._axon

    async def send_challenge(self, uid, axon: bt.AxonInfo, prompt: str) -> dict:
        """Send a challenge to a single axon and return the result."""
        dendrite = self.dendrite

        max_tokens=100
        temperature=0.5
        top_p=0.9
        completion=""

        synapse = InferenceSynapse(prompt=prompt, model='model', max_tokens=max_tokens, temperature=temperature, top_p=top_p, completion=completion)

        start_time = time.time()
        response = await dendrite.forward(
            axons=[axon],
            synapse=synapse,
            timeout=180.0,  # seconds
        )
        response_time_ms = int((time.time() - start_time) * 1000)

        # Determine success/failure
        success = response != [''] and response is not None
        
        # Log all responses if only_success is False
        if not only_success:
            bt.logging.info(f"RESPONSE FAILED: {response}")
        if success:
            bt.logging.info(f"RESPONSE SUCCESS: {response}")

        return {
            "uid": uid,
            "hotkey": axon.hotkey,
            "success": success,
            "response": response,
            "response_time_ms": response_time_ms,
        }

    # SEND CHALLENGE TASK ]
    # STAGE A [

    def stageA(self, port=8099) -> None:
        """Stage A: Initialize and start the bittensor node (merged stage1 and stage2)."""
        bt.logging.info(f"[green]Stage A: Initializing and starting the bittensor node.[/green]")
        bt.logging.info(f"Port: {port}")

        # Load keys
        self.load_keys()

        self._port = port

        # LOAD CONFIG (TEST)

        # CONFIG REFACTOR [

        # Store the original ValidatorConfig
        self.config_orig = self.config
        
        # Convert ValidatorConfig to bittensor config
        # This ensures SUBTENSOR_NETWORK and SUBTENSOR_ADDRESS are properly used
        from validator.config import validator_config_to_bittensor_config
        self.config = validator_config_to_bittensor_config(self.config_orig)
        
        bt.logging.info(f"Network configuration:")
        bt.logging.info(f"  - Network: {self.config.subtensor.network}")
        bt.logging.info(f"  - Chain Endpoint: {self.config.subtensor.chain_endpoint}")
        bt.logging.info(f"  - NetUID: {self.config.netuid}")

        config = self.config

        # CONFIG REFACTOR ]

        # CREATE SUBTENSOR 

        # The subtensor is our connection to the Bittensor blockchain.
        # Use the configured network and chain endpoint from environment variables
        bt.logging.info(f"Initializing subtensor with network: {self.config.subtensor.network}")
        self.subtensor = LooshSubnetSubtensor(
            network=self.config.subtensor.network, 
            config=self.config
        )
        
        # Setup config: This method determines the appropriate network and chain endpoint based on the provided network string or configuration object.
        self.chain_endpoint, self.network = self.subtensor.setup_config(self.config.network, self.config)
        
        bt.logging.info(f"Subtensor connected:")
        bt.logging.info(f"  - Network: {self.network}")
        bt.logging.info(f"  - Chain Endpoint: {self.chain_endpoint}")
        bt.logging.info(f"  - Subtensor: {self.subtensor}")

        # METAGRAPH

        self.updateMetagraph()

        # CREATE DENTDRITE

        # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
        dendrite = bt.dendrite(wallet=self.hotkey)
        self._dendrite = dendrite
        bt.logging.info(f"Dendrite: {self.dendrite}")

        bt.logging.info(f"Coldkey: {self.coldkey}")
        bt.logging.info(f"Hotkey: {self.hotkey}")

    # STAGE A ]
    # STAGE 3 [

    # TODO: refactor stage3 to setup axon?
        
    def stage3(self):
        port = self._port
        ip = '127.0.0.1'
        external_ip = ip
        external_port = None

        bt.logging.info(f"[green]Setup axon: {ip}:{port} {external_ip}:{external_port}[/green]")

        # Setup axon
        self.setup_axon(ip=ip, port=port, external_ip=external_ip, external_port=external_port)
        
        # Start the axon
        self.start_axon() 

#        self.axon.external_ip = '0.0.0.0'

        # Serve the axon (registers it with the subtensor)
        self.subtensor.serve_axon(netuid=self.config.netuid, axon=self.axon)

        #bt.logging.info(f"[green]Axon: {self.axon}[/green]")

    # STAGE 3 ]

    # Send challenge to multiple axons
    async def send_challenge_to_axons(self, axons: [bt.AxonInfo], prompt: str) -> list[dict]:
        """Send challenges to multiple axons and return results."""
        results = []
        try:
            for uid, axon in axons: 
                bt.logging.info(f"Challenge sending to {uid}/{axon.hotkey} - {axon.ip}:{axon.port}")
                result = await self.send_challenge(uid, axon, prompt)
                bt.logging.info(f"Challenge response: {result}")
                results.append(result)
        except Exception as e:
            bt.logging.error(f"Error sending challenge to axons: {str(e)}")
        return results

    # TODO: review and refactor 
    async def test_connection(self):
        axons = self.select_axons()
        for uid, axon in axons:
            try:
                bt.logging.info(f"Challenge sending to {uid}/{axon.hotkey} - {axon.ip}:{axon.port}")
                response = await self.send_challenge(uid, axon, "Hello, world!")
                bt.logging.info(f"Challenge response: {response}")
            except Exception as e:
                bt.logging.error(f"Error sending challenge to {uid}: {str(e)}")
                
    # TODO: review and refactor 
    async def test_dummy_connection(self):
        # DUMMYSYNAPSE
    
        # Step 5: Defincontinuee a dummy synapse (use appropriate Synapse for your subnet)
        synapse = bt.Synapse(
            # Add inputs depending on the subnet's expected Synapse fields
        )


        axon = self.axon
        dendrite = self.dendrite

        # Step 6: Make a forward call
        response = await dendrite.forward(
            axons=[axon],
            synapse=synapse,
            timeout=10.0,  # seconds
        )

        # Step 7: Inspect response
        print("Response:", response)


# BittensorNode ]

# Create BittensorNode instance
def create_node(config = None) -> BittensorNode:
    # Load configuration
    # TODO: remove validator config for tests
    config = config or get_validator_config()      
    
    bt.logging.info(f"Starting bittensor node with configuration:")
    bt.logging.info(f"Subnet: {config.netuid}")
    bt.logging.info(f"Network: {config.subtensor_network}")
    bt.logging.info(f"Wallet: {config.wallet_name}")
    bt.logging.info(f"Hotkey: {config.hotkey_name}")
    
    # Create and setup node
    return BittensorNode(config)


class LooshCell:
    
    def __init__(self, config=None):
        pass




    async def main_loop(self):
        """Main function to run the bittensor node."""
        
        node = create_node()

        try:
            # STAGE A
            node.stageA()

            # STAGE 3
            node.stage3()

            await node.test_dummy_connection()

            await node.test_connection()
            
            # MAIN LOOP - UNLIMITED [

            # Keep running
            bt.logging.info("Bittensor node is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

            # MAIN LOOP - UNLIMITED ]
              
        except KeyboardInterrupt:
            bt.logging.info("Received interrupt signal")
        except Exception as e:
            bt.logging.error(f"Error in main loop: {str(e)}")
        finally:
            node.stop()
            bt.logging.info("Bittensor node stopped")


if __name__ == "__main__":
    import asyncio
    try:
        bt.logging.info(f"Starting bittensor node with configuration:")
        cell = LooshCell()
        asyncio.run(cell.main_loop())
    except KeyboardInterrupt:
        pass
