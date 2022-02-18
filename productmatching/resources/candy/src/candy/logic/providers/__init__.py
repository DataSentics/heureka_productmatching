from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Candidate:
    id: str
    source: List[str]
    distance: Optional[float] = None
    relevance: Optional[float] = None
    data: Optional[dict] = None
