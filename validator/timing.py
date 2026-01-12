"""Timing tracking utility for pipeline performance monitoring."""

import time
import json
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict


# Pipeline stage names for consistency
class PipelineStages:
    """Standard pipeline stage names."""
    GATEWAY_RECEIVE = "gateway_receive"
    GATEWAY_CHALLENGE_CREATE = "gateway_challenge_create"
    CHALLENGE_API_RECEIVE = "challenge_api_receive"
    CHALLENGE_API_DB_INSERT = "challenge_api_db_insert"
    CHALLENGE_API_FLUVIO_PUBLISH = "challenge_api_fluvio_publish"
    CHALLENGE_PUSHER_CONSUME = "challenge_pusher_consume"
    CHALLENGE_PUSHER_PUSH = "challenge_pusher_push"
    VALIDATOR_RECEIVE = "validator_receive"
    VALIDATOR_SEND_TO_MINER = "validator_send_to_miner"
    MINER_INFERENCE = "miner_inference"
    MINER_RESPONSE = "miner_response"
    VALIDATOR_EVALUATION = "validator_evaluation"
    VALIDATOR_SEND_TO_API = "validator_send_to_api"
    CHALLENGE_API_RESPONSE_RECEIVE = "challenge_api_response_receive"
    CHALLENGE_API_RESPONSE_DB_INSERT = "challenge_api_response_db_insert"
    CHALLENGE_API_FLUVIO_NOTIFY = "challenge_api_fluvio_notify"
    GATEWAY_RESPONSE_RECEIVE = "gateway_response_receive"
    GATEWAY_RESPONSE_RETURN = "gateway_response_return"


@dataclass
class StageTiming:
    """Timing information for a single pipeline stage."""
    stage_name: str
    start_timestamp: float  # Unix timestamp
    end_timestamp: Optional[float] = None  # Unix timestamp
    elapsed_ms: Optional[float] = None  # Milliseconds
    
    def finish(self, end_time: Optional[float] = None) -> None:
        """Mark the stage as finished and calculate elapsed time."""
        if end_time is None:
            end_time = time.time()
        self.end_timestamp = end_time
        self.elapsed_ms = (end_time - self.start_timestamp) * 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "stage_name": self.stage_name,
            "start_timestamp": self.start_timestamp,
            "start_datetime": datetime.fromtimestamp(self.start_timestamp).isoformat(),
        }
        if self.end_timestamp is not None:
            result["end_timestamp"] = self.end_timestamp
            result["end_datetime"] = datetime.fromtimestamp(self.end_timestamp).isoformat()
        if self.elapsed_ms is not None:
            result["elapsed_ms"] = round(self.elapsed_ms, 2)
        return result


@dataclass
class PipelineTiming:
    """Complete timing information for a request pipeline."""
    correlation_id: str
    request_start_timestamp: float  # Unix timestamp when request started
    stages: List[StageTiming] = field(default_factory=list)
    total_elapsed_ms: Optional[float] = None
    
    def add_stage(self, stage_name: str, start_time: Optional[float] = None) -> StageTiming:
        """Add a new stage and return it for timing."""
        if start_time is None:
            start_time = time.time()
        stage = StageTiming(stage_name=stage_name, start_timestamp=start_time)
        self.stages.append(stage)
        return stage
    
    def finish_stage(self, stage_name: str, end_time: Optional[float] = None) -> Optional[StageTiming]:
        """Finish a stage by name."""
        # Find the most recent stage with this name that hasn't finished
        for stage in reversed(self.stages):
            if stage.stage_name == stage_name and stage.end_timestamp is None:
                stage.finish(end_time)
                return stage
        return None
    
    def finish(self, end_time: Optional[float] = None) -> None:
        """Finish the entire pipeline and calculate total elapsed time."""
        if end_time is None:
            end_time = time.time()
        
        # Finish any unfinished stages
        for stage in self.stages:
            if stage.end_timestamp is None:
                stage.finish(end_time)
        
        # Calculate total elapsed time
        self.total_elapsed_ms = (end_time - self.request_start_timestamp) * 1000.0
    
    def get_stage(self, stage_name: str) -> Optional[StageTiming]:
        """Get the most recent stage by name."""
        for stage in reversed(self.stages):
            if stage.stage_name == stage_name:
                return stage
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "correlation_id": self.correlation_id,
            "request_start_timestamp": self.request_start_timestamp,
            "request_start_datetime": datetime.fromtimestamp(self.request_start_timestamp).isoformat(),
            "stages": [stage.to_dict() for stage in self.stages],
        }
        if self.total_elapsed_ms is not None:
            result["total_elapsed_ms"] = round(self.total_elapsed_ms, 2)
            result["request_end_timestamp"] = self.request_start_timestamp + (self.total_elapsed_ms / 1000.0)
            result["request_end_datetime"] = datetime.fromtimestamp(result["request_end_timestamp"]).isoformat()
        return result
    
    def get_summary(self) -> str:
        """Get a human-readable summary of timing."""
        lines = [f"Pipeline timing for {self.correlation_id}:"]
        lines.append(f"  Total elapsed: {self.total_elapsed_ms:.2f}ms" if self.total_elapsed_ms else "  Total elapsed: N/A")
        lines.append("  Stages:")
        for stage in self.stages:
            elapsed_str = f"{stage.elapsed_ms:.2f}ms" if stage.elapsed_ms is not None else "N/A"
            lines.append(f"    - {stage.stage_name}: {elapsed_str}")
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineTiming':
        """Create a PipelineTiming instance from a dictionary."""
        # Reconstruct stages, filtering out datetime fields that are only for display
        stages = []
        for stage_data in data.get('stages', []):
            # Only pass fields that StageTiming.__init__ accepts
            stage_dict = {
                'stage_name': stage_data.get('stage_name'),
                'start_timestamp': stage_data.get('start_timestamp'),
                'end_timestamp': stage_data.get('end_timestamp'),
                'elapsed_ms': stage_data.get('elapsed_ms')
            }
            # Filter out None values to use defaults
            stage_dict = {k: v for k, v in stage_dict.items() if v is not None}
            stages.append(StageTiming(**stage_dict))
        
        instance = cls(
            correlation_id=data['correlation_id'],
            request_start_timestamp=data['request_start_timestamp'],
            stages=stages,
            total_elapsed_ms=data.get('total_elapsed_ms')
        )
        return instance
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineTiming':
        """Create a PipelineTiming instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
