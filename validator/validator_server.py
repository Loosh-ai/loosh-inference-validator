import os
import asyncio
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

from validator.config import ValidatorConfig
from validator.endpoints.availability import router as availability_router
from validator.endpoints.set_running import router as set_running_router
from validator.endpoints.get_stats import router as get_stats_router
from validator.endpoints.register import router as register_router
from validator.endpoints.challenges import router as challenges_router
    
def get_config():
    return ValidatorConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from validator.main import main_loop
    main_loop_task = asyncio.create_task(main_loop())
    yield
    # Shutdown
    main_loop_task.cancel()
    try:
        await main_loop_task
    except asyncio.CancelledError:
        pass

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Add dependencies
app.dependency_overrides[ValidatorConfig] = get_config

# API

# /set_running
app.include_router(
    set_running_router,
    prefix="/set_running",
    tags=["set_running"]
)

# /get_stats
app.include_router(
    get_stats_router,
    prefix="/get_stats",
    tags=["get_stats"]
)

# AVAILABILITY
app.include_router(
    availability_router,
    prefix="/availability",
    tags=["availability"]
)

# REGISTER
app.include_router(
    register_router,
    prefix="/register",
    tags=["register"]
)

# CHALLENGES
app.include_router(
    challenges_router,
    prefix="/challenges",
    tags=["challenges"]
)


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port
    )
