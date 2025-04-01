from pydantic import BaseModel
from typing import List

class Paper(BaseModel):
    id: str
    title: str
    authors: List[str]
    summary: str
    updated: str