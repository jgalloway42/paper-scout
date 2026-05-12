"""All DB read/write operations. The only module that writes to the database."""

import json
import os
import sqlite3
from datetime import date
from typing import Any

import numpy as np

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        db_path = os.environ.get("PAPER_SCOUT_DB_PATH", "paper_scout.db")
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
    return _conn


def set_connection(conn: sqlite3.Connection) -> None:
    """Override the module-level connection. Used by tests to inject tmp_db."""
    global _conn
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _conn = conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def insert_items(items: list[dict]) -> list[int]:
    """Insert items, skipping duplicates (by url). Returns list of inserted ids."""
    conn = _get_conn()
    inserted_ids: list[int] = []
    with conn:
        for item in items:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO items
                        (source, title, url, abstract, authors, published_date,
                         topic_bucket, summary, embedding, raw_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["source"],
                        item["title"],
                        item["url"],
                        item.get("abstract", ""),
                        json.dumps(item.get("authors", [])),
                        item["published_date"],
                        item["topic_bucket"],
                        item.get("summary"),
                        item.get("embedding"),
                        item["raw_id"],
                    ),
                )
                inserted_ids.append(cur.lastrowid)
            except sqlite3.IntegrityError:
                pass  # duplicate url — skip
    return inserted_ids


def get_all_embeddings() -> tuple[list[int], np.ndarray]:
    """Return (ids, (N, 384) float32 array). Returns empty array if no embeddings."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, embedding FROM items WHERE embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return [], np.empty((0, 384), dtype=np.float32)
    ids = [r["id"] for r in rows]
    vectors = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32).reshape(384) for r in rows]
    )
    return ids, vectors


def get_rated_embeddings(rating: str) -> np.ndarray:
    """Return (N, 384) float32 array for items with given rating ('up' or 'down')."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT i.embedding FROM items i
        JOIN ratings r ON r.item_id = i.id
        WHERE r.rating = ? AND i.embedding IS NOT NULL
        """,
        (rating,),
    ).fetchall()
    if not rows:
        return np.empty((0, 384), dtype=np.float32)
    return np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32).reshape(384) for r in rows]
    )


def get_item_by_id(item_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_dict(row) if row else None


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def insert_digest(week_start: date, items: list[dict]) -> int:
    """Insert digest + all items in a single transaction. Returns digest_id.

    Each item dict must have keys: item_id, relevance_score, is_wildcard, position.
    """
    conn = _get_conn()
    with conn:
        cur = conn.execute(
            "INSERT INTO digests (week_start) VALUES (?)",
            (week_start.isoformat(),),
        )
        digest_id = cur.lastrowid
        conn.executemany(
            """
            INSERT INTO digest_items
                (digest_id, item_id, relevance_score, is_wildcard, position)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    digest_id,
                    it["item_id"],
                    it["relevance_score"],
                    it["is_wildcard"],
                    it["position"],
                )
                for it in items
            ],
        )
    return digest_id


def get_latest_digest() -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM digests ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_digest_by_id(digest_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM digests WHERE id = ?", (digest_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_digest_history(limit: int = 10) -> list[dict]:
    """Return digests with their items (joined). Newest first."""
    conn = _get_conn()
    digests = conn.execute(
        "SELECT * FROM digests ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    result = []
    for d in digests:
        d_dict = _row_to_dict(d)
        items = conn.execute(
            """
            SELECT i.*, di.relevance_score, di.is_wildcard, di.position,
                   r.rating
            FROM digest_items di
            JOIN items i ON i.id = di.item_id
            LEFT JOIN ratings r ON r.item_id = i.id
            WHERE di.digest_id = ?
            ORDER BY di.position
            """,
            (d["id"],),
        ).fetchall()
        d_dict["items"] = [_row_to_dict(it) for it in items]
        result.append(d_dict)
    return result


def digest_exists(week_start: date) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM digests WHERE week_start = ?", (week_start.isoformat(),)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


def upsert_rating(item_id: int, rating: str) -> None:
    """INSERT OR REPLACE — one rating per paper."""
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO ratings (item_id, rating) VALUES (?, ?)",
            (item_id, rating),
        )


def get_all_ratings() -> list[tuple[dict, str]]:
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT i.*, r.rating FROM items i
        JOIN ratings r ON r.item_id = i.id
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    return [(_row_to_dict(r), r["rating"]) for r in rows]


def get_rated_items_with_summaries() -> list[tuple[dict, str]]:
    """Return (item_dict, rating) for all rated items that have a summary."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT i.id, i.title, i.summary, i.source, i.url, r.rating
        FROM items i
        JOIN ratings r ON r.item_id = i.id
        WHERE i.summary IS NOT NULL
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    return [(_row_to_dict(r), r["rating"]) for r in rows]


def count_ratings_since_last_profile() -> int:
    """Count ratings whose row ID is greater than the max rating ID at profile creation time.

    We approximate "ratings after profile" by counting ratings with id >
    the count of ratings that existed when the profile was generated
    (stored as rated_items_count). Returns total count if no profile exists.
    """
    conn = _get_conn()
    last_profile = conn.execute(
        "SELECT rated_items_count FROM profiles ORDER BY id DESC LIMIT 1"
    ).fetchone()
    total = conn.execute("SELECT COUNT(*) AS cnt FROM ratings").fetchone()["cnt"]
    if last_profile is None:
        return total
    return max(0, total - last_profile["rated_items_count"])


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def insert_profile(prose: str, mode: str, rated_count: int, provider: str) -> int:
    conn = _get_conn()
    with conn:
        cur = conn.execute(
            """
            INSERT INTO profiles (prose, mode, rated_items_count, llm_provider)
            VALUES (?, ?, ?, ?)
            """,
            (prose, mode, rated_count, provider),
        )
    return cur.lastrowid


def get_latest_profile() -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM profiles ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_profile_history(limit: int = 10) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM profiles ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Topic exposure
# ---------------------------------------------------------------------------


def get_topic_exposure(lookback_weeks: int) -> dict[str, int]:
    """Return {topic_bucket: seen_count} for buckets seen in the last N weeks."""
    from datetime import timedelta

    conn = _get_conn()
    cutoff = (date.today() - timedelta(weeks=lookback_weeks)).isoformat()
    rows = conn.execute(
        """
        SELECT topic_bucket, seen_count FROM topic_exposure
        WHERE last_seen >= ?
        """,
        (cutoff,),
    ).fetchall()
    return {r["topic_bucket"]: r["seen_count"] for r in rows}


def update_topic_exposure(
    buckets_seen: list[str], ratings: dict[str, str]
) -> None:
    """Increment seen_count for each bucket. Increment up/down counts from ratings.

    ratings: {topic_bucket: "up"|"down"} for buckets that were rated this digest.
    """
    conn = _get_conn()
    today = date.today().isoformat()
    with conn:
        for bucket in buckets_seen:
            up_inc = 1 if ratings.get(bucket) == "up" else 0
            down_inc = 1 if ratings.get(bucket) == "down" else 0
            conn.execute(
                """
                INSERT INTO topic_exposure (topic_bucket, seen_count, up_count, down_count, last_seen)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(topic_bucket) DO UPDATE SET
                    seen_count = seen_count + 1,
                    up_count   = up_count + excluded.up_count,
                    down_count = down_count + excluded.down_count,
                    last_seen  = excluded.last_seen
                """,
                (bucket, up_inc, down_inc, today),
            )
