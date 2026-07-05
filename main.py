from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from core.database import get_db, create_tables
from core.auth import hash_password, verify_password, create_access_token
from core.dependencies import get_current_user
from models.models import User, Task, TaskStatus, TaskCategory, EnergyLevel
from schemas.schemas import RegisterRequest, LoginRequest, TokenResponse, UserRead
from services.task_service import create_task, get_task, list_tasks, transition_task, get_task_history
from services.nudge_service import evaluate_and_log
from ml.ml_features import extract_features
from ml.ml_train import train, predict_completion
from ml.lgbm_train import train_procrastination, predict_procrastination_risk
from services.analytics_service import build_behavior_profile, get_behavior_profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    print("Tables ready!")
    yield


app = FastAPI(title="AURA", lifespan=lifespan)


# ── Auth ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AURA is alive"}


@app.post("/register", response_model=UserRead, status_code=201)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email           = payload.email,
        username        = payload.username,
        hashed_password = hash_password(payload.password),
        full_name       = payload.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    user.id = str(user.id)
    return user


@app.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


# ── Tasks ──────────────────────────────────────────────────────────────────────

@app.post("/tasks", status_code=201)
async def api_create_task(
    title              : str,
    category           : TaskCategory,
    energy_requirement : EnergyLevel,
    estimated_duration : int,
    priority           : int = 5,
    deadline           : datetime | None = None,
    db                 : AsyncSession = Depends(get_db),
    current_user       : User = Depends(get_current_user),
):
    task = await create_task(
        user_id            = current_user.id,
        title              = title,
        category           = category,
        energy_requirement = energy_requirement,
        estimated_duration = estimated_duration,
        priority           = priority,
        deadline           = deadline,
        db                 = db,
    )
    return {
        "id"    : str(task.id),
        "title" : task.title,
        "status": task.status,
    }


@app.get("/tasks")
async def api_list_tasks(
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    tasks = await list_tasks(current_user.id, db)
    return [
        {
            "id"      : str(t.id),
            "title"   : t.title,
            "status"  : t.status,
            "priority": t.priority,
            "category": t.category,
        }
        for t in tasks
    ]


@app.patch("/tasks/{task_id}/transition")
async def api_transition_task(
    task_id      : str,
    to_status    : TaskStatus,
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    try:
        task = await transition_task(
            task_id   = uuid.UUID(task_id),
            user_id   = current_user.id,
            to_status = to_status,
            db        = db,
        )
        return {
            "id"                   : str(task.id),
            "title"                : task.title,
            "status"               : task.status,
            "procrastination_count": task.procrastination_count,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/tasks/{task_id}/history")
async def api_task_history(
    task_id      : str,
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    try:
        events = await get_task_history(
            task_id = uuid.UUID(task_id),
            user_id = current_user.id,
            db      = db,
        )
        return [
            {
                "event_type" : e.event_type,
                "from_status": e.from_status,
                "to_status"  : e.to_status,
                "hour_of_day": e.hour_of_day,
                "day_of_week": e.day_of_week,
                "occurred_at": e.occurred_at,
            }
            for e in events
        ]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/nudge/evaluate")
async def api_nudge_evaluate(
    heart_rate          : int,
    hrv_ms              : float,
    app_switches        : int   = 0,
    screen_time_hours   : float = 0.0,
    minutes_since_break : int   = 0,
    db                  : AsyncSession = Depends(get_db),
    current_user        : User = Depends(get_current_user),
):
    result = await evaluate_and_log(
        user_id             = current_user.id,
        heart_rate          = heart_rate,
        hrv_ms              = hrv_ms,
        app_switches        = app_switches,
        screen_time_hours   = screen_time_hours,
        minutes_since_break = minutes_since_break,
        db                  = db,
    )
    return result


@app.post("/ml/train")
async def api_train(
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    df = await extract_features(db)

    if df.empty:
        return {
            "status" : "not enough data",
            "message": "Complete or abandon at least 10 tasks first",
        }

    metrics = train(df)
    return {"status": "trained", "metrics": metrics}


@app.post("/ml/predict")
async def api_predict(
    task_id      : str,
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    task = await get_task(uuid.UUID(task_id), current_user.id, db)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    category_map = {
        "deep_work": 0, "admin": 1, "learning": 2,
        "meeting": 3, "personal": 4, "health": 5,
    }
    energy_map = {
        "very_low": 0, "low": 1, "medium": 2, "high": 3, "peak": 4,
    }

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    features = {
        "estimated_duration"    : task.estimated_duration,
        "priority"              : task.priority,
        "category_enc"          : category_map.get(task.category.value, 0),
        "energy_requirement_enc": energy_map.get(task.energy_requirement.value, 2),
        "procrastination_count" : task.procrastination_count,
        "reschedule_count"      : task.reschedule_count,
        "skip_count"            : task.skip_count,
        "hour_of_day"           : now.hour,
        "day_of_week"           : now.weekday(),
    }

    prob = predict_completion(features)
    return {
        "task_id"               : task_id,
        "title"                 : task.title,
        "completion_probability": prob,
        "interpretation"        : (
            "likely to complete" if prob >= 0.6
            else "at risk of procrastination" if prob <= 0.4
            else "uncertain"
        ),
    }

@app.post("/ml/train/procrastination")
async def api_train_procrastination(
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    df = await extract_features(db)

    if df.empty:
        return {
            "status" : "not enough data",
            "message": "Complete or abandon at least 10 tasks first",
        }

    metrics = train_procrastination(df)
    return {"status": "trained", "metrics": metrics}


@app.post("/ml/predict/procrastination")
async def api_predict_procrastination(
    task_id      : str,
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    task = await get_task(uuid.UUID(task_id), current_user.id, db)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    category_map = {
        "deep_work": 0, "admin": 1, "learning": 2,
        "meeting"  : 3, "personal": 4, "health": 5,
    }
    energy_map = {
        "very_low": 0, "low": 1, "medium": 2, "high": 3, "peak": 4,
    }

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    features = {
        "estimated_duration"      : task.estimated_duration,
        "priority"                : task.priority,
        "category_enc"            : category_map.get(task.category.value, 0),
        "energy_requirement_enc"  : energy_map.get(task.energy_requirement.value, 2),
        "procrastination_count"   : task.procrastination_count,
        "reschedule_count"        : task.reschedule_count,
        "skip_count"              : task.skip_count,
        "hour_of_day"             : now.hour,
        "day_of_week"             : now.weekday(),
    }

    risk = predict_procrastination_risk(features)

    return {
        "task_id"               : task_id,
        "title"                 : task.title,
        "procrastination_risk"  : risk,
        "interpretation"        : (
            "high risk — likely to delay"    if risk >= 0.6
            else "low risk — likely to start" if risk <= 0.3
            else "moderate risk"
        ),
        "suggestion": (
            "Break this task into smaller pieces" if risk >= 0.6
            else "Good time to schedule this"     if risk <= 0.3
            else "Set a specific start time"
        ),
    }

@app.post("/analytics/profile/build")
async def api_build_profile(
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    profile = await build_behavior_profile(current_user.id, db)
    return {
        "user_id"                      : str(profile.user_id),
        "total_tasks_created"          : profile.total_tasks_created,
        "total_tasks_completed"        : profile.total_tasks_completed,
        "total_tasks_abandoned"        : profile.total_tasks_abandoned,
        "completion_rate"              : profile.completion_rate,
        "avg_estimation_error_minutes" : profile.avg_estimation_error_minutes,
        "avg_procrastination_count"    : profile.avg_procrastination_count,
        "avg_reschedule_count"         : profile.avg_reschedule_count,
        "avg_skip_count"               : profile.avg_skip_count,
        "peak_focus_hour_start"        : profile.peak_focus_hour_start,
        "peak_focus_hour_end"          : profile.peak_focus_hour_end,
        "most_procrastinated_category" : profile.most_procrastinated_category,
        "last_updated"                 : profile.last_updated,
    }


@app.get("/analytics/profile")
async def api_get_profile(
    db           : AsyncSession = Depends(get_db),
    current_user : User = Depends(get_current_user),
):
    profile = await get_behavior_profile(current_user.id, db)
    if profile is None:
        return {
            "message": "No profile yet. Hit POST /analytics/profile/build first."
        }
    return {
        "user_id"                      : str(profile.user_id),
        "total_tasks_created"          : profile.total_tasks_created,
        "total_tasks_completed"        : profile.total_tasks_completed,
        "total_tasks_abandoned"        : profile.total_tasks_abandoned,
        "completion_rate"              : profile.completion_rate,
        "avg_estimation_error_minutes" : profile.avg_estimation_error_minutes,
        "avg_procrastination_count"    : profile.avg_procrastination_count,
        "avg_reschedule_count"         : profile.avg_reschedule_count,
        "avg_skip_count"               : profile.avg_skip_count,
        "peak_focus_hour_start"        : profile.peak_focus_hour_start,
        "peak_focus_hour_end"          : profile.peak_focus_hour_end,
        "most_procrastinated_category" : profile.most_procrastinated_category,
        "last_updated"                 : profile.last_updated,
    }

@app.post("/ml/retrain")
async def api_retrain(
    current_user: User = Depends(get_current_user),
):
    """Trigger model retraining via Celery worker."""
    from workers.tasks import retrain_models
    task = retrain_models.delay()
    return {"status": "queued", "task_id": task.id}


@app.post("/ml/update-profiles")
async def api_update_profiles(
    current_user: User = Depends(get_current_user),
):
    """Trigger behavior profile update via Celery worker."""
    from workers.tasks import update_behavior_profiles
    task = update_behavior_profiles.delay()
    return {"status": "queued", "task_id": task.id}


@app.get("/ml/task-status/{task_id}")
async def api_task_status(
    task_id      : str,
    current_user : User = Depends(get_current_user),
):
    """Check status of a queued Celery task."""
    from workers.celery_app import celery_app
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status" : task.status,
        "result" : task.result if task.ready() else None,
    }