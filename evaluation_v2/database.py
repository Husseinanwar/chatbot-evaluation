import csv
import io
import os
from collections import defaultdict
from functools import lru_cache
from statistics import mean
from typing import Any

try:
    from supabase import Client, create_client
except ImportError:
    Client = Any
    create_client = None


RATINGS_TABLE = "ratings"
PAGE_SIZE = 1000


class SupabaseDatabaseError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    if create_client is None:
        raise SupabaseDatabaseError(
            "Das Paket 'supabase' ist nicht installiert. "
            "Bitte fuehre 'pip install -r requirements.txt' aus."
        )

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise SupabaseDatabaseError(
            "Supabase ist nicht konfiguriert. Bitte setze die Environment "
            "Variablen SUPABASE_URL und SUPABASE_KEY."
        )

    return create_client(supabase_url, supabase_key)


def ensure_database_ready() -> None:
    try:
        (
            get_supabase_client()
            .table(RATINGS_TABLE)
            .select("id")
            .limit(1)
            .execute()
        )
    except SupabaseDatabaseError:
        raise
    except Exception as error:
        raise SupabaseDatabaseError(
            "Keine Verbindung zur Supabase-Tabelle 'ratings' moeglich. "
            "Bitte pruefe SUPABASE_URL, SUPABASE_KEY, Tabellenname, "
            "RLS-Policies und ob die Tabelle bereits per SQL erstellt wurde."
        ) from error


def save_rating(participant_id: str, question_data: dict, rating: int) -> None:
    payload = {
        "participant_id": participant_id,
        "question_id": question_data["question_id"],
        "question": question_data["question"],
        "model": question_data["model"],
        "rating": rating
    }

    try:
        (
            get_supabase_client()
            .table(RATINGS_TABLE)
            .upsert(
                payload,
                on_conflict="participant_id,question_id,model"
            )
            .execute()
        )
    except SupabaseDatabaseError:
        raise
    except Exception as error:
        raise SupabaseDatabaseError(
            "Die Bewertung konnte nicht in Supabase gespeichert werden."
        ) from error


def fetch_ratings() -> list[dict]:
    ensure_database_ready()
    ratings: list[dict] = []
    start = 0

    while True:
        try:
            response = (
                get_supabase_client()
                .table(RATINGS_TABLE)
                .select(
                    "id,participant_id,question_id,question,model,rating,created_at"
                )
                .order("created_at", desc=True)
                .order("id", desc=True)
                .range(start, start + PAGE_SIZE - 1)
                .execute()
            )
        except Exception as error:
            raise SupabaseDatabaseError(
                "Bewertungen konnten nicht aus Supabase geladen werden."
            ) from error

        batch = response.data or []

        for row in batch:
            item = dict(row)
            item["timestamp"] = item.get("created_at") or ""
            ratings.append(item)

        if len(batch) < PAGE_SIZE:
            break

        start += PAGE_SIZE

    return ratings


def fetch_statistics(total_questions: int) -> dict:
    ratings = fetch_ratings()
    total_ratings = len(ratings)
    participant_ids = {row["participant_id"] for row in ratings}
    overall_average = mean_rating([row["rating"] for row in ratings])

    ratings_by_model: dict[str, list[int]] = defaultdict(list)
    ratings_by_question: dict[tuple[str, str], list[int]] = defaultdict(list)
    questions_by_participant: dict[str, set[str]] = defaultdict(set)

    for row in ratings:
        rating = int(row["rating"])
        ratings_by_model[row["model"]].append(rating)
        ratings_by_question[(row["question_id"], row["question"])].append(rating)
        questions_by_participant[row["participant_id"]].add(row["question_id"])

    model_stats = sorted(
        [
            {
                "model": model,
                "count": len(values),
                "average": mean(values)
            }
            for model, values in ratings_by_model.items()
        ],
        key=lambda row: (-row["average"], row["model"])
    )
    question_stats = sorted(
        [
            {
                "question_id": question_id,
                "question": question,
                "count": len(values),
                "average": mean(values)
            }
            for (question_id, question), values in ratings_by_question.items()
        ],
        key=lambda row: question_sort_key(row["question_id"])
    )
    completed_participants = sum(
        1
        for question_ids in questions_by_participant.values()
        if total_questions > 0 and len(question_ids) == total_questions
    )

    return {
        "total_ratings": total_ratings,
        "participant_count": len(participant_ids),
        "overall_average": overall_average,
        "model_stats": model_stats,
        "question_stats": question_stats,
        "best_model": model_stats[0]["model"] if model_stats else None,
        "worst_model": model_stats[-1]["model"] if model_stats else None,
        "completed_participants": completed_participants
    }


def build_ratings_csv() -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "participant_id",
            "question_id",
            "question",
            "model",
            "rating",
            "created_at"
        ]
    )

    for row in fetch_ratings():
        writer.writerow(
            [
                row["participant_id"],
                row["question_id"],
                row["question"],
                row["model"],
                row["rating"],
                row.get("created_at", "")
            ]
        )

    return output.getvalue()


def mean_rating(values: list[int]) -> float | None:
    if not values:
        return None

    return mean(values)


def question_sort_key(question_id: str) -> tuple[int, str]:
    if question_id.isdigit():
        return int(question_id), question_id

    return 10**9, question_id
