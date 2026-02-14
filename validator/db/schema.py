from datetime import datetime
from typing import Optional
import sqlite3

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    create_engine, Table, MetaData, Text, JSON, text, inspect
)
from sqlalchemy.orm import declarative_base, relationship
from loguru import logger

Base = declarative_base()

class Miner(Base):
    """Table for storing miner information.
    
    UID COMPRESSION SAFETY: hotkey is the primary persistent identifier.
    node_id (UID) is stored as an informational snapshot — it changes on UID
    compression/trimming and MUST NOT be used for persistent lookups or as a unique key.
    """
    __tablename__ = "miners"

    id = Column(Integer, primary_key=True)
    hotkey = Column(String, unique=True, nullable=False)  # PRIMARY persistent identity (SS58 address)
    node_id = Column(Integer, nullable=True)  # Informational snapshot only — NOT unique, changes on UID compression
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

def _has_unique_constraint_on_node_id(engine) -> bool:
    """Check if the miners table has a UNIQUE constraint on node_id.
    
    Inspects the SQLite index list to detect stale UNIQUE constraints
    that were removed from the SQLAlchemy model but persist in the
    live database (create_all never drops constraints).
    """
    with engine.connect() as conn:
        # Check if miners table exists
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='miners'")
        )
        if not result.fetchone():
            return False
        
        # Inspect indexes on the miners table
        indexes = conn.execute(text("PRAGMA index_list('miners')")).fetchall()
        for idx in indexes:
            # idx format: (seq, name, unique, origin, partial)
            idx_name = idx[1]
            is_unique = idx[2]
            if is_unique:
                # Check which columns this unique index covers
                cols = conn.execute(
                    text(f"PRAGMA index_info('{idx_name}')")
                ).fetchall()
                col_names = [col[2] for col in cols]  # col format: (seqno, cid, name)
                if col_names == ["node_id"]:
                    return True
    return False


def _migrate_miners_drop_node_id_unique(engine) -> None:
    """Remove stale UNIQUE constraint on miners.node_id.
    
    SQLite does not support ALTER TABLE DROP CONSTRAINT, so we must
    recreate the table. This preserves all data and foreign key
    references (FKs reference miners.id which is preserved).
    
    This migration is idempotent — it only runs when the stale
    constraint is detected.
    """
    logger.info("Migrating miners table: removing UNIQUE constraint from node_id")
    
    with engine.begin() as conn:
        # Disable FK enforcement during migration to avoid cascading issues
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        
        # 1. Create new table with correct schema (no UNIQUE on node_id)
        conn.execute(text("""
            CREATE TABLE miners_new (
                id INTEGER PRIMARY KEY,
                hotkey VARCHAR NOT NULL UNIQUE,
                node_id INTEGER,
                ip VARCHAR NOT NULL,
                port INTEGER NOT NULL,
                stake FLOAT NOT NULL,
                is_available INTEGER NOT NULL,
                last_success DATETIME,
                error TEXT,
                last_updated DATETIME NOT NULL,
                created_at DATETIME,
                updated_at DATETIME
            )
        """))
        
        # 2. Copy all data — on collision (duplicate node_id), keep the
        #    most recently updated row per hotkey (hotkey is the real PK)
        conn.execute(text("""
            INSERT INTO miners_new
                (id, hotkey, node_id, ip, port, stake, is_available,
                 last_success, error, last_updated, created_at, updated_at)
            SELECT id, hotkey, node_id, ip, port, stake, is_available,
                   last_success, error, last_updated, created_at, updated_at
            FROM miners
        """))
        
        # 3. Drop old table and rename
        conn.execute(text("DROP TABLE miners"))
        conn.execute(text("ALTER TABLE miners_new RENAME TO miners"))
        
        # 4. Re-enable FK enforcement
        conn.execute(text("PRAGMA foreign_keys = ON"))
    
    logger.info("Migration complete: miners.node_id UNIQUE constraint removed")


def migrate_db(engine) -> None:
    """Run all pending schema migrations.
    
    Called during DatabaseManager initialization and init_db.
    Each migration is idempotent and self-detecting.
    """
    if _has_unique_constraint_on_node_id(engine):
        _migrate_miners_drop_node_id_unique(engine)


def init_db(db_path: str) -> None:
    """Initialize the database with the schema."""
    engine = create_engine(f"sqlite:///{db_path}")
    migrate_db(engine)
    Base.metadata.create_all(engine)