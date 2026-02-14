"""
Application entry point for Water Watch backend.

This file is intentionally THIN.
- No business logic
- No alert logic
- No SMTP / Redis / DB logic

Purpose:
- Bootstrap the application
- Mount API routes
- Manage lifecycle hooks

FastAPI is used CURRENTLY, but this file is structured
so the framework can be replaced later.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# API routes
from api.routes.alerts import router as alerts_router

# Infrastructure services (for lifecycle)
from services.smtp_service import get_smtp_service
from services.redis_throttle import get_redis_client
from services.audit_logger import get_audit_logger


# --------------------------------------------------
# App Factory (important for future scalability)
# --------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Water Watch Alert Engine",
        description="Critical water contamination alert system",
        version="0.1.0",
    )

    # -------------------------
    # Middleware
    # -------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------
    # Routes
    # -------------------------
    app.include_router(
        alerts_router,
        prefix="/alerts",
        tags=["alerts"],
    )

    # -------------------------
    # Health check (non-negotiable)
    # -------------------------
    @app.get("/health", tags=["system"])
    def health_check():
        return {
            "status": "ok",
            "service": "water-watch-backend",
        }

    return app


# --------------------------------------------------
# Lifecycle Hooks
# --------------------------------------------------

app = create_app()


@app.on_event("startup")
def on_startup():
    """
    Initialize long-lived services.
    """
    get_smtp_service()
    get_redis_client()
    get_audit_logger()

    print("✅ Water Watch backend started")


@app.on_event("shutdown")
def on_shutdown():
    """
    Graceful shutdown hooks (future-proofing).
    """
    print("🛑 Water Watch backend shutting down")
