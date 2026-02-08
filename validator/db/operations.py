from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from validator.db.schema import (
    Base, Miner, InferenceChallenge, InferenceResponse,
    EvaluationResult, MinerScore, SybilDetectionResult
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
        """Log or update miner information.
        
        UID COMPRESSION SAFETY: Looks up miners by HOTKEY (persistent SS58 address),
        not by node_id (UID). UIDs are transient indices that change on UID compression.
        node_id is stored as an informational snapshot and updated each observation.
        """
        with self.get_session() as session:
            # Look up by HOTKEY (persistent identity), not node_id (UID)
            miner = session.query(Miner).filter_by(hotkey=hotkey).first()
            
            if miner:
                # Update existing miner — including node_id which may change on UID compression
                miner.node_id = node_id  # Informational snapshot, updated each observation
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
    
    def log_sybil_detection_result(
        self,
        challenge_id: Optional[int],
        suspicious_pairs: List[Dict[str, Any]],
        suspicious_groups: List[Dict[str, Any]],
        analysis_report: str,
        high_similarity_threshold: float,
        very_high_similarity_threshold: float
    ) -> None:
        """Log sybil detection results for analysis."""
        with self.get_session() as session:
            result = SybilDetectionResult(
                challenge_id=challenge_id,
                suspicious_pairs_count=len(suspicious_pairs),
                suspicious_groups_count=len(suspicious_groups),
                suspicious_pairs=suspicious_pairs,
                suspicious_groups=suspicious_groups,
                analysis_report=analysis_report,
                high_similarity_threshold=high_similarity_threshold,
                very_high_similarity_threshold=very_high_similarity_threshold
            )
            session.add(result)
            session.commit()
    
    def delete_sybil_detection_result(self, record_id: int) -> bool:
        """Delete a sybil detection result by ID after successful submission."""
        try:
            with self.get_session() as session:
                record = session.query(SybilDetectionResult).filter(
                    SybilDetectionResult.id == record_id
                ).first()
                if record:
                    session.delete(record)
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting sybil detection result {record_id}: {e}", exc_info=True)
            return False
    
    def get_miner_ema_scores(
        self,
        lookback_hours: int = 24,
        alpha: float = 0.3
    ) -> Dict[str, float]:
        """
        Calculate EMA (Exponential Moving Average) scores for miners based on emissions from evaluations.
        
        The EMA gives more weight to recent performance while still considering historical data.
        Formula: EMA = alpha * current + (1 - alpha) * previous_EMA
        
        UID COMPRESSION SAFETY: Returns dict keyed by miner HOTKEY (persistent SS58 address),
        NOT by UID. During transition, handles both old UID-keyed and new hotkey-keyed emissions
        by mapping UIDs to hotkeys via the Miner table.
        
        Args:
            lookback_hours: How far back to look for evaluation results (default: 24 hours)
            alpha: EMA smoothing factor (0-1). Higher values weight recent data more.
                   Default 0.3 means 30% weight to current, 70% to history.
        
        Returns:
            Dict mapping miner_hotkey (str) -> ema_score (float)
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        with self.get_session() as session:
            # Query evaluation results from lookback period, ordered by time ascending
            # (oldest first for proper EMA calculation)
            results = (
                session.query(EvaluationResult)
                .filter(EvaluationResult.created_at >= cutoff_time)
                .order_by(EvaluationResult.created_at.asc())
                .all()
            )
            
            if not results:
                logger.info(f"No evaluation results found in last {lookback_hours} hours for EMA calculation")
                return {}
            
            # Build UID->hotkey mapping from Miner table for backward compatibility
            # (old emissions are keyed by UID strings, new ones by hotkey)
            miners = session.query(Miner).all()
            uid_to_hotkey = {str(m.node_id): m.hotkey for m in miners if m.node_id is not None}
            
            # Track EMA scores per miner (keyed by HOTKEY)
            ema_scores: Dict[str, float] = {}
            
            for result in results:
                if not result.emissions:
                    continue
                
                # Emissions dict has format: {"key": emission_value, ...}
                # key is either a hotkey (new format) or a UID string (old format)
                for emission_key, emission_value in result.emissions.items():
                    try:
                        emission = float(emission_value)
                        
                        # Determine the hotkey for this emission entry
                        # Try UID->hotkey mapping first (for old UID-keyed emissions)
                        hotkey = uid_to_hotkey.get(emission_key)
                        if hotkey is None:
                            # Key is not a known UID — assume it's already a hotkey (new format)
                            hotkey = emission_key
                        
                        if hotkey in ema_scores:
                            # Update EMA: alpha * current + (1 - alpha) * previous
                            ema_scores[hotkey] = alpha * emission + (1 - alpha) * ema_scores[hotkey]
                        else:
                            # First observation for this miner - initialize with current value
                            ema_scores[hotkey] = emission
                            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid emission data for key {emission_key}: {e}")
                        continue
            
            logger.info(
                f"Calculated EMA scores for {len(ema_scores)} miners from "
                f"{len(results)} evaluations (lookback: {lookback_hours}h, alpha: {alpha})"
            )
            
            return ema_scores
    
    def get_miner_last_success_times(self) -> Dict[str, datetime]:
        """
        Get the last successful response timestamp for each miner.
        
        Used by weight setting to apply a freshness gate - miners without
        recent successful responses should not receive weight.
        
        UID COMPRESSION SAFETY: Returns dict keyed by miner HOTKEY (persistent SS58 address),
        NOT by UID. UIDs are transient indices that change on UID compression/trimming.
        
        Returns:
            Dict mapping miner_hotkey (str) -> last_success (datetime)
        """
        with self.get_session() as session:
            miners = (
                session.query(Miner)
                .filter(Miner.last_success.isnot(None))
                .all()
            )
            
            result = {
                miner.hotkey: miner.last_success
                for miner in miners
                if miner.last_success is not None and miner.hotkey
            }
            
            logger.debug(f"Retrieved last_success times for {len(result)} miners")
            return result
    
    def cleanup_old_data(self, retention_hours: int = 48) -> None:
        """
        Clean up old challenge, response, and evaluation records.
        
        Retains data for retention_hours (default 48h = 2x EMA lookback of 24h).
        This ensures sufficient history for EMA calculation while preventing
        unbounded database growth.
        
        NOTE: Sybil detection records are NOT cleaned up here - they are managed
        by SybilSyncTask which deletes records after successful submission to
        the Challenge API.
        
        Args:
            retention_hours: Hours of history to retain (default: 48)
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
        
        with self.get_session() as session:
            try:
                # Delete old evaluation results first (depends on challenges)
                deleted_evaluations = session.query(EvaluationResult).filter(
                    EvaluationResult.created_at < cutoff_time
                ).delete(synchronize_session='fetch')
                
                # Delete old inference responses (depends on challenges and miners)
                deleted_responses = session.query(InferenceResponse).filter(
                    InferenceResponse.created_at < cutoff_time
                ).delete(synchronize_session='fetch')
                
                # Delete old challenges
                deleted_challenges = session.query(InferenceChallenge).filter(
                    InferenceChallenge.created_at < cutoff_time
                ).delete(synchronize_session='fetch')
                
                # Delete old miner scores
                deleted_scores = session.query(MinerScore).filter(
                    MinerScore.created_at < cutoff_time
                ).delete(synchronize_session='fetch')
                
                session.commit()
                
                logger.info(
                    f"Database cleanup completed: "
                    f"deleted {deleted_evaluations} evaluations, "
                    f"{deleted_responses} responses, "
                    f"{deleted_challenges} challenges, "
                    f"{deleted_scores} scores "
                    f"(retention: {retention_hours}h, sybil records managed by SybilSyncTask)"
                )
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error during database cleanup: {e}", exc_info=True)
                raise