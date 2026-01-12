from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    create_engine, Table, MetaData, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Miner(Base):
    """Table for storing miner information."""
    __tablename__ = "miners"

    id = Column(Integer, primary_key=True)
    hotkey = Column(String, unique=True, nullable=False)
    node_id = Column(Integer, unique=True, nullable=False)
    ip = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    stake = Column(Float, nullable=False)
    is_available = Column(Integer, nullable=False)  # 0=False, 1=True
    last_success = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
    last_updated = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class InferenceChallenge(Base):
    """Table for storing inference challenges."""
    __tablename__ = "inference_challenges"

    id = Column(Integer, primary_key=True)
    prompt = Column(Text, nullable=False)
    model = Column(String, nullable=False)
    max_tokens = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    top_p = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class InferenceResponse(Base):
    """Table for storing inference responses from miners."""
    __tablename__ = "inference_responses"

    id = Column(Integer, primary_key=True)
    challenge_id = Column(Integer, ForeignKey("inference_challenges.id"), nullable=False)
    miner_id = Column(Integer, ForeignKey("miners.id"), nullable=False)
    response_text = Column(Text, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    challenge = relationship("InferenceChallenge")
    miner = relationship("Miner")

class EvaluationResult(Base):
    """Table for storing evaluation results."""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True)
    challenge_id = Column(Integer, ForeignKey("inference_challenges.id"), nullable=False)
    consensus_score = Column(Float, nullable=False)
    heatmap_path = Column(String, nullable=True)
    narrative = Column(Text, nullable=True)
    emissions = Column(JSON, nullable=True)  # Store emissions as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    
    challenge = relationship("InferenceChallenge")

class MinerScore(Base):
    """Table for storing miner scores."""
    __tablename__ = "miner_scores"

    id = Column(Integer, primary_key=True)
    miner_id = Column(Integer, ForeignKey("miners.id"), nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    miner = relationship("Miner")

class SybilDetectionResult(Base):
    """Table for storing sybil detection results."""
    __tablename__ = "sybil_detection_results"

    id = Column(Integer, primary_key=True)
    challenge_id = Column(Integer, ForeignKey("inference_challenges.id"), nullable=True)
    suspicious_pairs_count = Column(Integer, nullable=False, default=0)
    suspicious_groups_count = Column(Integer, nullable=False, default=0)
    suspicious_pairs = Column(JSON, nullable=True)  # Store pairs as JSON
    suspicious_groups = Column(JSON, nullable=True)  # Store groups as JSON
    analysis_report = Column(Text, nullable=True)
    high_similarity_threshold = Column(Float, nullable=False)
    very_high_similarity_threshold = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    challenge = relationship("InferenceChallenge")

def init_db(db_path: str) -> None:
    """Initialize the database with the schema."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine) 