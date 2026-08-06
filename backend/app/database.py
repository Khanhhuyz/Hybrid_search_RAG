"""
Database setup — SQLite via SQLAlchemy (async).
Stores document metadata, chunk metadata, and processing status.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, Enum as SAEnum, text
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
    index_version = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    progress_stage = Column(String, default="queued", nullable=False)
    progress_current = Column(Integer, default=0, nullable=False)
    progress_total = Column(Integer, default=0, nullable=False)
    heartbeat_at  = Column(DateTime, nullable=True)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ChunkModel(Base):
    __tablename__ = "chunks"

    id          = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    content     = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    page_end    = Column(Integer, nullable=True)
    section     = Column(String, nullable=True)
    parent_id   = Column(String, nullable=True, index=True)
    chunk_type  = Column(String, default="text", nullable=False)
    parent_content = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)
    char_start  = Column(Integer, nullable=True)
    char_end    = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    embedded    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AgentRunModel(Base):
    __tablename__ = "agent_runs"

    id = Column(String, primary_key=True)
    request = Column(Text, nullable=False)
    intent = Column(String, nullable=False)
    status = Column(String, nullable=False, default="running")
    plan_json = Column(Text, nullable=False, default="[]")
    steps_json = Column(Text, nullable=False, default="[]")
    answer = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    artifact_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    preview = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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
        # create_all does not add columns to an existing SQLite database.
        columns = await conn.execute(text("PRAGMA table_info(documents)"))
        existing = {row[1] for row in columns.fetchall()}
        migrations = {
            "progress_stage": "ALTER TABLE documents ADD COLUMN progress_stage VARCHAR NOT NULL DEFAULT 'queued'",
            "progress_current": "ALTER TABLE documents ADD COLUMN progress_current INTEGER NOT NULL DEFAULT 0",
            "progress_total": "ALTER TABLE documents ADD COLUMN progress_total INTEGER NOT NULL DEFAULT 0",
            "heartbeat_at": "ALTER TABLE documents ADD COLUMN heartbeat_at DATETIME",
            "index_version": "ALTER TABLE documents ADD COLUMN index_version INTEGER NOT NULL DEFAULT 0",
        }
        for name, statement in migrations.items():
            if name not in existing:
                await conn.execute(text(statement))
        chunk_columns = await conn.execute(text("PRAGMA table_info(chunks)"))
        existing_chunks = {row[1] for row in chunk_columns.fetchall()}
        chunk_migrations = {
            "page_end": "ALTER TABLE chunks ADD COLUMN page_end INTEGER",
            "parent_id": "ALTER TABLE chunks ADD COLUMN parent_id VARCHAR",
            "chunk_type": "ALTER TABLE chunks ADD COLUMN chunk_type VARCHAR NOT NULL DEFAULT 'text'",
            "parent_content": "ALTER TABLE chunks ADD COLUMN parent_content TEXT",
            "metadata_json": "ALTER TABLE chunks ADD COLUMN metadata_json TEXT",
        }
        for name, statement in chunk_migrations.items():
            if name not in existing_chunks:
                await conn.execute(text(statement))
