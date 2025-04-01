from app.models.paper import Paper
from typing import List

# This will be replaced with actual API calls later
async def get_latest_updates() -> List[Paper]:
    # Mock data for now
    return [
        
        Paper(
            id="2105.67890",
            title="Transformer Models for Natural Language Processing",
            authors=["Brown, K.", "Lee, M."],
            summary="A comprehensive review of transformer-based models in NLP...",
            updated="2023-03-28"
        ),
        Paper(
            id="2105.54321",
            title="Reinforcement Learning Applications",
            authors=["Garcia, R.", "Wilson, T."],
            summary="This study demonstrates practical applications of reinforcement learning...",
            updated="2023-03-27"
        )
    ]