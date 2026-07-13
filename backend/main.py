from fastapi import FastAPI

from api.router import router
from config.settings import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )
    app.include_router(router)
    return app


app = create_app()
