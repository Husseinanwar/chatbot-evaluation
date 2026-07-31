import csv
import os
import random
import uuid
from hmac import compare_digest
from pathlib import Path
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from dotenv import load_dotenv
load_dotenv()

from database import (
    SupabaseDatabaseError,
    build_ratings_csv,
    ensure_database_ready,
    fetch_ratings,
    fetch_statistics,
    save_rating
)


BASE_DIR = Path(__file__).resolve().parent
SOURCE_RESULTS_FILE = BASE_DIR / "data" / "answers.csv"
EXPECTED_MODELS = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b", "mistral"]
SESSION_COOKIE = "evaluation_v2_session"
ADMIN_SESSION_COOKIE = "evaluation_v2_admin"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

app = FastAPI(title="Evaluation der Chatbot-Antworten")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
markdown_renderer = MarkdownIt("commonmark", {"html": False, "breaks": True})

SESSIONS: dict[str, dict] = {}
ADMIN_SESSIONS: set[str] = set()


def render_markdown_answer(answer: str) -> str:
    return markdown_renderer.render(answer or "")


def question_template_context(
    question_data: dict,
    current_index: int,
    total_questions: int,
    selected_rating: int | None = None,
    error: str | None = None
) -> dict:
    return {
        "question_data": question_data,
        "answer_html": render_markdown_answer(question_data.get("answer", "")),
        "current_number": current_index + 1,
        "total_questions": total_questions,
        "progress_percent": int(((current_index + 1) / total_questions) * 100),
        "selected_rating": selected_rating,
        "is_first": current_index == 0,
        "is_last": current_index == total_questions - 1,
        "error": error
    }


def load_questions() -> list[dict]:
    if not SOURCE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Die Datei {SOURCE_RESULTS_FILE} wurde nicht gefunden. "
            "Bitte lege evaluation_v2/data/answers.csv mit den Fragen "
            "und Modellantworten ab."
        )

    with open(SOURCE_RESULTS_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"Nr", "Frage", "Modell", "Antwort"}

        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns - set(reader.fieldnames or []))
            raise ValueError(
                "evaluation_v2/data/answers.csv hat nicht das erwartete Format. "
                f"Fehlende Spalten: {', '.join(missing)}"
            )

        questions_by_id = {}
        question_order = []

        for row in reader:
            question_id = row.get("Nr", "").strip()
            question = row.get("Frage", "").strip()
            model = row.get("Modell", "").strip()
            answer = row.get("Antwort", "").strip()

            if not question_id or not question or not model or not answer:
                continue

            if model not in EXPECTED_MODELS:
                raise ValueError(
                    f"Unbekanntes Modell in evaluation_v2/data/answers.csv: {model}. "
                    "Erwartet werden llama3.2:3b, gemma2:2b, qwen2.5:3b und mistral."
                )

            if question_id not in questions_by_id:
                questions_by_id[question_id] = {
                    "question_id": question_id,
                    "question": question,
                    "answers": {}
                }
                question_order.append(question_id)

            existing_question = questions_by_id[question_id]["question"]

            if existing_question != question:
                raise ValueError(
                    f"Frage Nr. {question_id} hat unterschiedliche Fragetexte "
                    "in evaluation_v2/data/answers.csv."
                )

            if model in questions_by_id[question_id]["answers"]:
                raise ValueError(
                    f"Frage Nr. {question_id} enthaelt mehrere Antworten fuer {model}."
                )

            questions_by_id[question_id]["answers"][model] = answer

    questions = [
        questions_by_id[question_id]
        for question_id in question_order
    ]

    if not questions:
        raise ValueError(
            "evaluation_v2/data/answers.csv enthaelt keine auswertbaren Fragen."
        )

    for question in questions:
        missing_models = [
            model
            for model in EXPECTED_MODELS
            if model not in question["answers"]
        ]

        if missing_models:
            raise ValueError(
                f"Frage Nr. {question['question_id']} ist unvollstaendig. "
                "Fehlende Modelle: " + ", ".join(missing_models)
            )

    return questions


def create_participant_session() -> str:
    questions = load_questions()
    participant_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    question_order = list(range(len(questions)))
    random.shuffle(question_order)

    assigned_questions = []

    for index in question_order:
        question = questions[index]
        model = random.choice(EXPECTED_MODELS)
        assigned_questions.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "model": model,
                "answer": question["answers"][model],
                "rating": None
            }
        )

    SESSIONS[session_id] = {
        "participant_id": participant_id,
        "questions": assigned_questions,
        "current_index": 0,
        "completed": False
    }

    return session_id


def get_session(request: Request) -> dict | None:
    session_id = request.cookies.get(SESSION_COOKIE)

    if not session_id:
        return None

    return SESSIONS.get(session_id)


def is_admin_authenticated(request: Request) -> bool:
    admin_session = request.cookies.get(ADMIN_SESSION_COOKIE)
    return bool(admin_session and admin_session in ADMIN_SESSIONS)


def safe_next_path(next_path: str | None) -> str:
    if not next_path or not next_path.startswith("/") or next_path.startswith("//"):
        return "/admin"

    return next_path


def admin_redirect(request: Request) -> RedirectResponse:
    next_path = quote(str(request.url.path), safe="/")
    return RedirectResponse(url=f"/login?next={next_path}", status_code=303)


def render_error(request: Request, message: str) -> Response:
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "message": message
        },
        status_code=400
    )


@app.on_event("startup")
def startup() -> None:
    try:
        ensure_database_ready()
    except SupabaseDatabaseError as error:
        print(f"Supabase-Hinweis: {error}")


@app.get("/")
def index(request: Request) -> Response:
    return templates.TemplateResponse(
        request,
        "index.html",
        {}
    )


@app.get("/login")
def login_page(request: Request, next: str = "/admin") -> Response:
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "next_path": safe_next_path(next),
            "error": None
        }
    )


@app.post("/login")
async def login(request: Request) -> Response:
    form_data = parse_qs((await request.body()).decode("utf-8"))
    password = form_data.get("password", [""])[0]
    next_path = safe_next_path(form_data.get("next_path", ["/admin"])[0])

    if not compare_digest(password, ADMIN_PASSWORD):
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "next_path": next_path,
                "error": "Falsches Passwort."
            },
            status_code=401
        )

    admin_session = str(uuid.uuid4())
    ADMIN_SESSIONS.add(admin_session)
    response = RedirectResponse(url=next_path, status_code=303)
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=admin_session,
        httponly=True,
        samesite="lax"
    )
    return response


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    admin_session = request.cookies.get(ADMIN_SESSION_COOKIE)

    if admin_session:
        ADMIN_SESSIONS.discard(admin_session)

    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(ADMIN_SESSION_COOKIE)
    return response


@app.post("/start")
def start_evaluation(request: Request) -> Response:
    try:
        session_id = create_participant_session()
    except (FileNotFoundError, ValueError) as error:
        return render_error(request, str(error))

    response = RedirectResponse(url="/question", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax"
    )
    return response


@app.get("/question")
def question_page(request: Request) -> Response:
    session = get_session(request)

    if not session:
        return RedirectResponse(url="/", status_code=303)

    if session["completed"]:
        return RedirectResponse(url="/finished", status_code=303)

    current_index = session["current_index"]
    questions = session["questions"]
    question_data = questions[current_index]

    return templates.TemplateResponse(
        request,
        "question.html",
        question_template_context(
            question_data=question_data,
            current_index=current_index,
            total_questions=len(questions),
            selected_rating=question_data.get("rating")
        )
    )


@app.post("/question")
async def submit_question(request: Request) -> Response:
    session = get_session(request)

    if not session:
        return RedirectResponse(url="/", status_code=303)

    form_data = parse_qs((await request.body()).decode("utf-8"))
    action = form_data.get("action", [""])[0]
    rating_value = form_data.get("rating", [None])[0]
    rating = int(rating_value) if rating_value and rating_value.isdigit() else None

    current_index = session["current_index"]
    questions = session["questions"]
    question_data = questions[current_index]

    if action == "back":
        if current_index > 0:
            session["current_index"] -= 1
        return RedirectResponse(url="/question", status_code=303)

    if rating is None or rating < 1 or rating > 10:
        return templates.TemplateResponse(
            request,
            "question.html",
            question_template_context(
                question_data=question_data,
                current_index=current_index,
                total_questions=len(questions),
                selected_rating=rating,
                error="Bitte waehlen Sie eine Bewertung aus."
            ),
            status_code=400
        )

    question_data["rating"] = rating

    try:
        save_rating(session["participant_id"], question_data, rating)
    except SupabaseDatabaseError as error:
        return render_error(request, str(error))

    if action == "finish":
        session["completed"] = True
        return RedirectResponse(url="/finished", status_code=303)

    if current_index < len(questions) - 1:
        session["current_index"] += 1

    return RedirectResponse(url="/question", status_code=303)


@app.get("/finished")
def finished(request: Request) -> Response:
    session = get_session(request)

    if not session:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request,
        "finished.html",
        {}
    )


@app.get("/admin")
def admin(request: Request) -> Response:
    if not is_admin_authenticated(request):
        return admin_redirect(request)

    try:
        ratings = fetch_ratings()
    except SupabaseDatabaseError as error:
        return render_error(request, str(error))

    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "ratings": ratings
        }
    )


@app.get("/download")
def download(request: Request) -> Response:
    if not is_admin_authenticated(request):
        return admin_redirect(request)

    try:
        csv_content = build_ratings_csv()
    except SupabaseDatabaseError as error:
        return render_error(request, str(error))

    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=ratings_export.csv"
        }
    )


@app.get("/statistics")
def statistics(request: Request) -> Response:
    if not is_admin_authenticated(request):
        return admin_redirect(request)

    try:
        statistics_data = fetch_statistics(total_questions=len(load_questions()))
    except (FileNotFoundError, ValueError, SupabaseDatabaseError) as error:
        return render_error(request, str(error))

    return templates.TemplateResponse(
        request,
        "statistics.html",
        statistics_data
    )
