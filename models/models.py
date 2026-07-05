import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime, Integer, Float, Text, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import enum


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────────────────────

class TaskStatus(str, enum.Enum):
    DRAFT          = "draft"
    SCHEDULED      = "scheduled"
    IN_PROGRESS    = "in_progress"
    PAUSED         = "paused"
    COMPLETED      = "completed"
    PARTIALLY_DONE = "partially_done"
    POSTPONED      = "postponed"
    SKIPPED        = "skipped"
    ABANDONED      = "abandoned"


class TaskCategory(str, enum.Enum):
    DEEP_WORK = "deep_work"
    ADMIN     = "admin"
    LEARNING  = "learning"
    MEETING   = "meeting"
    PERSONAL  = "personal"
    HEALTH    = "health"


class EnergyLevel(str, enum.Enum):
    VERY_LOW = "very_low"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    PEAK     = "peak"


# ── User ───────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200))
    timezone: Mapped[str] = mapped_column(String(50), default="Asia/Kolkata")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# ── Task ───────────────────────────────────────────────────────────────────────

class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[TaskCategory] = mapped_column(
        SAEnum(TaskCategory), nullable=False, default=TaskCategory.DEEP_WORK
    )
    energy_requirement: Mapped[EnergyLevel] = mapped_column(
        SAEnum(EnergyLevel), nullable=False, default=EnergyLevel.MEDIUM
    )
    estimated_duration: Mapped[int] = mapped_column(Integer, nullable=False)  # minutes
    actual_duration: Mapped[int | None] = mapped_column(Integer)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # State machine
    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus), nullable=False, default=TaskStatus.DRAFT
    )
    procrastination_count: Mapped[int] = mapped_column(Integer, default=0)
    reschedule_count: Mapped[int] = mapped_column(Integer, default=0)
    skip_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TaskEvent(Base):
    __tablename__ = "task_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(50))
    to_status: Mapped[str | None] = mapped_column(String(50))
    reason: Mapped[str | None] = mapped_column(Text)

    # ML signals — captured at the moment of the event
    hour_of_day: Mapped[int | None] = mapped_column(Integer)
    day_of_week: Mapped[int | None] = mapped_column(Integer)
    stress_score: Mapped[int | None] = mapped_column(Integer)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

class StressLog(Base):
    __tablename__ = "stress_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    heart_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    hrv_ms: Mapped[float] = mapped_column(Float, nullable=False)
    app_switches: Mapped[int] = mapped_column(Integer, default=0)
    screen_time_hours: Mapped[float] = mapped_column(Float, default=0.0)
    minutes_since_break: Mapped[int] = mapped_column(Integer, default=0)

    # Computed
    stress_score: Mapped[int] = mapped_column(Integer, nullable=False)
    stress_level: Mapped[str] = mapped_column(String(20), nullable=False)
    nudge_type: Mapped[str] = mapped_column(String(30), nullable=False)

    hour_of_day: Mapped[int | None] = mapped_column(Integer)
    day_of_week: Mapped[int | None] = mapped_column(Integer)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class UserBehaviorProfile(Base):
    __tablename__ = "user_behavior_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    # Completion patterns
    total_tasks_created: Mapped[int] = mapped_column(Integer, default=0)
    total_tasks_completed: Mapped[int] = mapped_column(Integer, default=0)
    total_tasks_abandoned: Mapped[int] = mapped_column(Integer, default=0)
    completion_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Time accuracy
    avg_estimation_error_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    # positive = underestimated, negative = overestimated

    # Procrastination signals
    avg_procrastination_count: Mapped[float] = mapped_column(Float, default=0.0)
    avg_reschedule_count: Mapped[float] = mapped_column(Float, default=0.0)
    avg_skip_count: Mapped[float] = mapped_column(Float, default=0.0)

    # Peak focus hours (learned from completions)
    peak_focus_hour_start: Mapped[int | None] = mapped_column(Integer)
    peak_focus_hour_end: Mapped[int | None] = mapped_column(Integer)

    # Most procrastinated category
    most_procrastinated_category: Mapped[str | None] = mapped_column(String(50))

    # Timestamps
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )