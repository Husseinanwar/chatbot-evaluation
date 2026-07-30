import csv
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt


BASE_DIR = Path(__file__).resolve().parent
SOURCE_RESULTS_FILE = BASE_DIR / "data" / "answers.csv"
RATING_RESULTS_FILE = BASE_DIR / "results" / "results.csv"
EXPECTED_MODELS = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b", "mistral"]
SESSION_COOKIE = "evaluation_v2_session"

app = FastAPI(title="Evaluation der Chatbot-Antworten")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
markdown_renderer = MarkdownIt("commonmark", {"html": False, "breaks": True})

SESSIONS = {}


def render_markdown_answer(answer):
    return markdown_renderer.render(answer or "")


def question_template_context(
    question_data,
    current_index,
    total_questions,
    selected_rating=None,
    error=None
):
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


def load_questions():
    if not SOURCE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Die Datei {SOURCE_RESULTS_FILE} wurde nicht gefunden. "
            "Bitte fuehre zuerst die bestehende Evaluation aus, damit "
            "evaluation/results.csv vorhanden ist."
        )

    with open(SOURCE_RESULTS_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"Nr", "Frage", "Modell", "Antwort"}

        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns - set(reader.fieldnames or []))
            raise ValueError(
                "evaluation/results.csv hat nicht das erwartete Format. "
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
                    f"Unbekanntes Modell in evaluation/results.csv: {model}. "
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
                    "in evaluation/results.csv."
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
            "evaluation/results.csv enthaelt keine auswertbaren Fragen."
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


def create_participant_session():
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


def get_session(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)

    if not session_id:
        return None

    return SESSIONS.get(session_id)


def ensure_results_file():
    RATING_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RATING_RESULTS_FILE.exists():
        return

    with open(RATING_RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "participant_id",
                "question_id",
                "question",
                "model",
                "rating",
                "timestamp"
            ]
        )


def save_rating(participant_id, question_data, rating):
    ensure_results_file()

    rows = []

    if RATING_RESULTS_FILE.exists():
        with open(RATING_RESULTS_FILE, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

    timestamp = datetime.now(timezone.utc).isoformat()
    updated = False

    for row in rows:
        if (
            row["participant_id"] == participant_id
            and row["question_id"] == question_data["question_id"]
        ):
            row["question"] = question_data["question"]
            row["model"] = question_data["model"]
            row["rating"] = str(rating)
            row["timestamp"] = timestamp
            updated = True
            break

    if not updated:
        rows.append(
            {
                "participant_id": participant_id,
                "question_id": question_data["question_id"],
                "question": question_data["question"],
                "model": question_data["model"],
                "rating": str(rating),
                "timestamp": timestamp
            }
        )

    with open(RATING_RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "participant_id",
            "question_id",
            "question",
            "model",
            "rating",
            "timestamp"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_error(request: Request, message: str):
    return templates.TemplateResponse(
        request,
        "error.html",
        {
            "message": message
        },
        status_code=400
    )


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {}
    )


@app.post("/start")
def start_evaluation(request: Request):
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
def question_page(request: Request):
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
async def submit_question(request: Request):
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
    save_rating(session["participant_id"], question_data, rating)

    if action == "finish":
        session["completed"] = True
        return RedirectResponse(url="/finished", status_code=303)

    if current_index < len(questions) - 1:
        session["current_index"] += 1

    return RedirectResponse(url="/question", status_code=303)


@app.get("/finished")
def finished(request: Request):
    session = get_session(request)

    if not session:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request,
        "finished.html",
        {}
    )
