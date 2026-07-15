"""Route aggregation: imports every resource router and composes them into
the single ``router`` that ``api.py`` mounts under ``/v1``. A new route module
must be registered here to be served.
"""

from fastapi import APIRouter

from .agent_configs import router as agent_configs_router
from .connections import router as connections_router
from .conversations import router as conversations_router
from .cron import router as cron_router
from .events import router as events_router
from .evolution import router as evolution_router
from .files import router as files_router
from .gateway import router as gateway_router
from .mcp_servers import router as mcp_servers_router
from .memory import router as memory_router
from .organizations import router as organizations_router
from .runs import router as runs_router
from .skills import router as skills_router
from .system import router as system_router
from .teams import router as teams_router
from .traces import router as traces_router
from .usage import router as usage_router
from .users import router as users_router

router = APIRouter()
router.include_router(runs_router)
router.include_router(organizations_router)
router.include_router(conversations_router)
router.include_router(agent_configs_router)
router.include_router(teams_router)
router.include_router(skills_router)
router.include_router(evolution_router)
router.include_router(events_router)
router.include_router(usage_router)
router.include_router(users_router)
router.include_router(memory_router)
router.include_router(traces_router)
router.include_router(cron_router)
router.include_router(gateway_router)
router.include_router(system_router)
router.include_router(connections_router)
router.include_router(mcp_servers_router)
router.include_router(files_router)
