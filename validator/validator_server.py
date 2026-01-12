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
from validator.endpoints.fiber import router as fiber_router, fiber_server
from validator.network.fiber_server import FiberServer
    
def get_config():
    return ValidatorConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Initialize Fiber server
    from validator.config import get_validator_config
    config = get_validator_config()
    fiber_key_ttl = getattr(config, 'fiber_key_ttl_seconds', 3600)
    
    global fiber_server
    fiber_server_instance = FiberServer(key_ttl_seconds=fiber_key_ttl)
    # Set the global fiber_server in the endpoints module
    import validator.endpoints.fiber as fiber_endpoints_module
    fiber_endpoints_module.fiber_server = fiber_server_instance
    
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

# FIBER (MLTS secure communication)
app.include_router(
    fiber_router,
    prefix="/fiber",
    tags=["fiber"]
)


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port
    )
