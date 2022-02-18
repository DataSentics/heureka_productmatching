import pydantic
import enum
import typing as t


class Decision(enum.Enum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


class Match(pydantic.BaseModel):
    match: Decision
    details: str
    # xgboost confidence for easier handling, not included if decided by other checks
    confidence: t.Optional[float]
