import typing as t

from pydantic import BaseModel


class Product(BaseModel):
    name: str
    prices: t.Optional[t.List[float]]
    shops: t.Optional[t.List[int]]
    category_id: t.Optional[str] = ""
    attributes: t.Optional[t.Dict[str, str]] = {}
    eans: t.Optional[t.List[int]]
    ean_required: t.Optional[bool] = False
    unique_names: t.Optional[bool] = False


class Offer(BaseModel):
    name: str
    price: float
    shop: t.Union[str, int]
    attributes: t.Optional[t.Dict[str, t.Union[str, list, set]]] = {}
    parsed_attributes: t.Optional[t.Dict[str, t.Union[str, list, set]]] = {}
    ean: t.Optional[int] = None
