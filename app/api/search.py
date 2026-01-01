from fastapi import APIRouter
from app.services.search_service import search

router = APIRouter()


@router.get("/search")
def search_api(q: str, top_k: int = 5):
    return search(query=q, top_k=top_k)
