"""Pydantic models for API request/response."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool


class RateQueryParams(BaseModel):
    item_id: int
    rating: str
    token: str
