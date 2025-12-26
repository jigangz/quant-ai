from typing import List, Dict, Any
import json
from sqlalchemy import text
from app.db.engine import engine


def upsert_features(rows: List[Dict[str, Any]]) -> None:
    """
    Upsert feature rows into features table.

    Each row format:
    {
      "ticker": str,
      "date": date,
      "features": dict   # will be stored as JSONB
    }
    """
    if not rows:
        return

    sql = text(
        """
        insert into features (ticker, date, features)
        values (:ticker, :date, :features)
        on conflict (ticker, date)
        do update set features = excluded.features
        """
    )

    # key：Python dict -> JSON string
    payload = [
        {
            "ticker": r["ticker"],
            "date": r["date"],
            "features": json.dumps(r["features"]),
        }
        for r in rows
    ]

    with engine.begin() as conn:
        conn.execute(sql, payload)


def get_features(ticker: str, limit: int = 100):
    """
    Fetch latest feature rows for a ticker.
    """
    sql = text(
        """
        select ticker, date, features
        from features
        where ticker = :ticker
        order by date desc
        limit :limit
        """
    )

    with engine.begin() as conn:
        result = conn.execute(
            sql,
            {"ticker": ticker, "limit": limit},
        )

        rows = []
        for row in result:
            item = dict(row._mapping)

            # 🔁 JSON string -> Python dict
            if isinstance(item.get("features"), str):
                item["features"] = json.loads(item["features"])

            rows.append(item)

        return rows
