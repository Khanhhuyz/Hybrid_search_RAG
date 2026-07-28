"""
Database setup — SQLite via SQLAlchemy (async).
Stores document metadata, chunk metadata, and processing status.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, Enum as SAEnum
from datetime import datetime, timezone
import enum

from app.config import settings

# Convert sqlite:/// → sqlite+aiosqlite:///
DATABASE_URL = settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(DATABASE_URL, echo=settings.DEBUG, connect_args={"timeout": 30})
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


# ─── Enums ───────────────────────────────────────────────────────────────────

class ProcessingStatus(str, enum.Enum):
    PENDING   = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


# ─── Models ──────────────────────────────────────────────────────────────────

class DocumentModel(Base):
    __tablename__ = "documents"

    id            = Column(String, primary_key=True)
    filename      = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    file_type     = Column(String, nullable=False)
    file_size     = Column(Integer, nullable=False)          # bytes
    file_path     = Column(String, nullable=False)
    status        = Column(SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False)
    chunk_count   = Column(Integer, default=0)
    entity_count  = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ChunkModel(Base):
    __tablename__ = "chunks"

    id          = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    content     = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    section     = Column(String, nullable=True)
    char_start  = Column(Integer, nullable=True)
    char_end    = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    embedded    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─── Dependency ──────────────────────────────────────────────────────────────

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
