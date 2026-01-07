from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()

class User(Base):
    """Table for storing user information."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    hotkey = Column(String, unique=True, nullable=False)
    is_available = Column(Integer, nullable=False)  # 0=False, 1=True
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_users_db(db_path: str) -> None:
    """Initialize the users database with the schema."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

class UserDatabaseManager:
    """Manager for user database operations."""
    
    def __init__(self, db_path: str):
        """Initialize user database connection."""
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()
    
    def update_user(self, hotkey: str, is_available: bool) -> None:
        """Create or update user record."""
        with self.get_session() as session:
            # Check if user exists
            user = session.query(User).filter_by(hotkey=hotkey).first()
            
            if user:
                # Update existing user
                user.is_available = 1 if is_available else 0
                user.updated_at = datetime.utcnow()
            else:
                # Create new user
                user = User(
                    hotkey=hotkey,
                    is_available=1 if is_available else 0
                )
                session.add(user)
            
            session.commit()
    
    def get_user(self, hotkey: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by hotkey."""
        with self.get_session() as session:
            user = session.query(User).filter_by(hotkey=hotkey).first()
            
            if user:
                return {
                    "id": user.id,
                    "hotkey": user.hotkey,
                    "is_available": bool(user.is_available),
                    "created_at": user.created_at,
                    "updated_at": user.updated_at
                }
            return None
