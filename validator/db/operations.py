from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker, Session

from validator.db.schema import (
    Base, Miner, InferenceChallenge, InferenceResponse,
    EvaluationResult, MinerScore
)
from validator.challenge.challenge_types import InferenceResponse as ChallengeResponse

class DatabaseManager:
    """Manager for database operations."""
    
    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()
    
    def log_miner(
        self, 
        node_id: int, 
        hotkey: str, 
        ip: str, 
        port: int, 
        stake: float,
        is_available: bool,
        last_success: Optional[datetime] = None,
        error: Optional[str] = None
    ) -> None:
        """Log or update miner information."""
        with self.get_session() as session:
            # Check if miner exists
            miner = session.query(Miner).filter_by(node_id=node_id).first()
            
            if miner:
                # Update existing miner
                miner.hotkey = hotkey
                miner.ip = ip
                miner.port = port
                miner.stake = stake
                miner.last_updated = datetime.utcnow()
                
                # Update new fields if provided
                miner.is_available = 1 if is_available else 0
                if last_success is not None:
                    miner.last_success = last_success
                if error is not None:
                    miner.error = error
            else:
                # Create new miner
                miner = Miner(
                    node_id=node_id,
                    hotkey=hotkey,
                    ip=ip,
                    port=port,
                    stake=stake,
                    is_available=1 if is_available else 0,
                    last_success=last_success,
                    error=error,
                    last_updated=datetime.utcnow()
                )
                session.add(miner)
            
            session.commit()
    
    def create_challenge(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> InferenceChallenge:
        """Create a new inference challenge."""
        with self.get_session() as session:
            challenge = InferenceChallenge(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            session.add(challenge)
            session.commit()
            return challenge
    
    def log_inference_response(
        self,
        challenge_id: int,
        miner_id: int,
        response: ChallengeResponse
    ) -> None:
        """Log an inference response from a miner."""
        with self.get_session() as session:
            db_response = InferenceResponse(
                challenge_id=challenge_id,
                miner_id=miner_id,
                response_text=response.response_text,
                response_time_ms=response.response_time_ms
            )
            session.add(db_response)
            session.commit()
    
    def log_evaluation_result(
        self,
        challenge_id: int,
        consensus_score: float,
        heatmap_path: Optional[str],
        narrative: Optional[str],
        emissions: Dict[str, float]
    ) -> None:
        """Log evaluation results for a challenge."""
        with self.get_session() as session:
            result = EvaluationResult(
                challenge_id=challenge_id,
                consensus_score=consensus_score,
                heatmap_path=heatmap_path,
                narrative=narrative,
                emissions=emissions
            )
            session.add(result)
            
            # Mark challenge as completed
            session.execute(
                update(InferenceChallenge)
                .where(InferenceChallenge.id == challenge_id)
                .values(completed_at=datetime.utcnow())
            )
            
            session.commit()
    
    def update_miner_score(self, miner_id: int, score: float) -> None:
        """Update a miner's score."""
        with self.get_session() as session:
            miner_score = MinerScore(
                miner_id=miner_id,
                score=score
            )
            session.add(miner_score)
            session.commit()
    
    def get_miner_scores(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent miner scores."""
        with self.get_session() as session:
            scores = (
                session.query(MinerScore)
                .order_by(MinerScore.created_at.desc())
                .limit(limit)
                .all()
            )
            
            return [
                {
                    "miner_id": score.miner_id,
                    "score": score.score,
                    "created_at": score.created_at
                }
                for score in scores
            ] 