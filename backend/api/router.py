from fastapi import APIRouter

from schemas.health import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(message="HireScore Backend Running")
