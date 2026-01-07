import pytest
import asyncio

# TODO: review

from unittest.mock import Mock, patch
from validator.network.InferenceSynapse import (
    InferenceSynapse, 
    inference, 
    blacklist, 
    priority
)
from validator.network.bittensor_node import (
    BittensorNode
)


class TestInferenceSynapse:
    """Test the InferenceSynapse class."""
    
    def test_synapse_creation(self):
        """Test synapse creation with valid data."""
        synapse = InferenceSynapse(
            prompt="Test prompt",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        
        assert synapse.prompt == "Test prompt"
        assert synapse.model == "test-model"
        assert synapse.max_tokens == 100
        assert synapse.temperature == 0.7
        assert synapse.top_p == 0.95
        assert synapse.completion is None
    
    def test_deserialize_method(self):
        """Test deserialize method returns completion."""
        synapse = InferenceSynapse(
            prompt="Test prompt",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        synapse.completion = "Test completion"
        
        result = synapse.deserialize()
        assert result == "Test completion"
    
    def test_required_hash_fields(self):
        """Test required_hash_fields property."""
        synapse = InferenceSynapse(
            prompt="Test prompt",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        
        assert synapse.required_hash_fields == ['prompt']


class TestCoreFunctions:
    """Test the core functions: inference, blacklist, priority."""
    
    def test_inference_function(self):
        """Test inference function returns synapse with completion."""
        synapse = InferenceSynapse(
            prompt="Test prompt",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        
        result = inference(synapse)
        
        assert result == synapse
        assert result.completion == "I am a bittensor inference node"
    
    def test_blacklist_function(self):
        """Test blacklist function always returns (False, "")."""
        synapse = InferenceSynapse(
            prompt="Test prompt",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        
        is_blacklisted, reason = blacklist(synapse)
        
        assert is_blacklisted is False
        assert reason == ""
    
    def test_priority_function(self):
        """Test priority function always returns 0.0."""
        synapse = InferenceSynapse(
            prompt="Test prompt",
            model="test-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )
        
        priority_value = priority(synapse)
        
        assert priority_value == 0.0


class TestBittensorNode:
    """Test the BittensorNode class."""
    
    @patch('validator.network.bittensor_node.get_validator_config')
    def test_node_initialization(self, mock_config):
        """Test node initialization with config."""
        mock_config.return_value = Mock()
        mock_config.return_value.wallet_name = "test_wallet"
        mock_config.return_value.hotkey_name = "test_hotkey"
        
        node = BittensorNode()
        
        assert node.config is not None
        assert node.axon is None
        assert node.hotkey is None
        assert node.coldkey is None
    
    @patch('validator.network.bittensor_node.load_hotkey_keypair')
    @patch('validator.network.bittensor_node.load_coldkeypub_keypair')
    def test_load_keys(self, mock_load_coldkey, mock_load_hotkey):
        """Test key loading using fiber utilities."""
        mock_hotkey = Mock()
        mock_coldkey = Mock()
        mock_load_hotkey.return_value = mock_hotkey
        mock_load_coldkey.return_value = mock_coldkey
        
        config = Mock()
        config.wallet_name = "test_wallet"
        config.hotkey_name = "test_hotkey"
        
        node = BittensorNode(config)
        node.load_keys()
        
        assert node.hotkey == mock_hotkey
        assert node.coldkey == mock_coldkey
        mock_load_hotkey.assert_called_once_with("test_wallet", "test_hotkey")
        mock_load_coldkey.assert_called_once_with("test_wallet")
    
    @patch('validator.network.bittensor_node.bt.axon')
    def test_setup_axon(self, mock_axon_class):
        """Test axon setup."""
        mock_axon = Mock()
        mock_axon_class.return_value = mock_axon
        
        node = BittensorNode()
        node.setup_axon(port=8099)
        
        assert node.axon == mock_axon
        mock_axon_class.assert_called_once_with(port=8099)
        mock_axon.attach.assert_called_once_with(
            forward_fn=inference,
            blacklist_fn=blacklist,
            priority_fn=priority
        )
    
    def test_start_without_axon(self):
        """Test start method raises error when axon not setup."""
        node = BittensorNode()
        
        with pytest.raises(RuntimeError, match="Axon not setup"):
            node.start()
    
    @patch('validator.network.bittensor_node.bt.axon')
    def test_start_and_stop(self, mock_axon_class):
        """Test start and stop methods."""
        mock_axon = Mock()
        mock_axon_class.return_value = mock_axon
        
        node = BittensorNode()
        node.setup_axon()
        node.start()
        node.stop()
        
        mock_axon.start.assert_called_once()
        mock_axon.stop.assert_called_once()


class TestIntegration:
    """Integration tests with mock bittensor."""
    
    @patch('validator.network.bittensor_node.bt.axon')
    @patch('validator.network.bittensor_node.load_hotkey_keypair')
    @patch('validator.network.bittensor_node.load_coldkeypub_keypair')
    def test_full_node_setup(self, mock_load_coldkey, mock_load_hotkey, mock_axon_class):
        """Test full node setup and operation."""
        mock_hotkey = Mock()
        mock_coldkey = Mock()
        mock_axon = Mock()
        
        mock_load_hotkey.return_value = mock_hotkey
        mock_load_coldkey.return_value = mock_coldkey
        mock_axon_class.return_value = mock_axon
        
        config = Mock()
        config.wallet_name = "test_wallet"
        config.hotkey_name = "test_hotkey"
        
        node = BittensorNode(config)
        node.load_keys()
        node.setup_axon(port=8099)
        node.start()
        node.stop()
        
        # Verify all methods were called
        mock_load_hotkey.assert_called_once_with("test_wallet", "test_hotkey")
        mock_load_coldkey.assert_called_once_with("test_wallet")
        mock_axon_class.assert_called_once_with(port=8099)
        mock_axon.attach.assert_called_once()
        mock_axon.start.assert_called_once()
        mock_axon.stop.assert_called_once()
    
    def test_synapse_communication(self):
        """Test synapse creation and processing."""
        # Create synapse
        synapse = InferenceSynapse(
            prompt="Hello, how are you?",
            model="test-model",
            max_tokens=50,
            temperature=0.8,
            top_p=0.9
        )
        
        # Process through inference function
        result = inference(synapse)
        
        # Verify result
        assert result.completion == "I am a bittensor inference node"
        assert result.prompt == "Hello, how are you?"
        
        # Test deserialization
        completion = result.deserialize()
        assert completion == "I am a bittensor inference node"
