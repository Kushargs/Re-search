from fastapi import APIRouter, HTTPException
from app.services.arxiv_service import get_latest_updates
from app.models.paper import Paper
from typing import List

router = APIRouter()

@router.get("/updates", response_model=List[Paper])
async def read_updates():
    try:
        updates = await get_latest_updates()
        return updates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))