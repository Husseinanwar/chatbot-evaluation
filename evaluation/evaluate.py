import csv
import sys
import time    
from pathlib import Path

import psutil    

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import backend    


QUESTIONS_FILE = ROOT / "evaluation" / "questions.txt"
RESULT_FILE = ROOT / "evaluation" / "results.csv"

backend.BENCHMARK_MODE = True    
MODELS = backend.AVAILABLE_MODELS    

TOP_K = 3
RERANKING = False


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def format_sources(sources):
    formatted_sources = []

    for source in sources:
        formatted_sources.append(
            f"{source['source']} "
            f"(S.{source['page']})"    
        )

    return "; ".join(formatted_sources)


def ram_percent():    
    return round(psutil.virtual_memory().percent, 1)


def main():
    questions = load_questions()

    print(f"{len(questions)} Fragen geladen.")
    print(f"Ergebnisse werden gespeichert in: {RESULT_FILE}")
    print("Alte Ergebnisse werden ersetzt.")
    print(f"Ollama Num Thread: {backend.OLLAMA_NUM_THREAD}")    

    with open(RESULT_FILE, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "Nr",
            "Frage",
            "Modell",
            "Antwort",
            "Quellen",
            "Chroma (s)",
            "BM25 (s)",
            "Re-Ranking (s)",
            "LLM (s)",
            "LLM Load (s)",    
            "Prompt Eval (s)",    
            "Token Eval (s)",    
            "Prompt Tokens",    
            "Output Tokens",    
            "Num Predict",    
            "Num Ctx",    
            "Num Thread",    
            "Gesamt (s)",
            "RAM Vor (%)",    
            "RAM Nach (%)"    
        ])

        for model in MODELS:    
            ram_before_model = ram_percent()    
            model_rows = []    

            print(f"\nModell: {model}")    
            print(f"RAM vor Modelldurchlauf: {ram_before_model:.1f} %")    

            try:    
                for nr, question in enumerate(questions, start=1):    
                    print(f"  Frage {nr}: {question}")    

                    result = backend.ask_question(    
                        frage=question,
                        model=model,
                        top_k=TOP_K,
                        reranking=RERANKING
                    )

                    timings = result["timings"]    

                    model_rows.append([    
                        nr,
                        question,
                        model,
                        result["answer"],
                        format_sources(result["sources"]),
                        round(timings["chroma"], 3),
                        round(timings["bm25"], 3),
                        round(timings["reranking"], 3),
                        round(timings["llm"], 3),
                        round(timings.get("llm_load", 0.0), 3),    
                        round(timings.get("llm_prompt_eval", 0.0), 3),    
                        round(timings.get("llm_token_eval", 0.0), 3),    
                        timings.get("llm_prompt_tokens", 0),    
                        timings.get("llm_generated_tokens", 0),    
                        timings.get("llm_num_predict", 0),    
                        timings.get("llm_num_ctx", 0),    
                        backend.OLLAMA_NUM_THREAD,    
                        round(timings["total"], 3)
                    ])
            finally:    
                unloaded = backend.unload_model(model)    
                time.sleep(2)    
                ram_after_model = ram_percent()    
                print(f"Modell entladen: {model} ({'ok' if unloaded else 'fehlgeschlagen'})")    
                print(f"RAM nach Modelldurchlauf: {ram_after_model:.1f} %")    

            for row in model_rows:    
                writer.writerow(row + [ram_before_model, ram_after_model])    

            csvfile.flush()    

    print("\nEvaluation abgeschlossen.")
    print(f"Ergebnis gespeichert unter: {RESULT_FILE}")


if __name__ == "__main__":
    main()
