PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup corpus index eval eval-retrieval app test docker clean all

setup:
	python3 -m venv .venv && $(PIP) install -q -U pip && $(PIP) install -q -r requirements.txt

corpus:          ## download + chunk the AI Act
	$(PY) -m src.euactrag.fetch
	$(PY) -m src.euactrag.ingest

index:           ## embed + build the vector store
	$(PY) -m src.euactrag.index

eval:            ## full evaluation (needs GROQ_API_KEY)
	$(PY) eval/run_eval.py

eval-retrieval:  ## retrieval metrics only, no API key needed
	$(PY) eval/run_eval.py --no-generation --tag retrieval_only

report:          ## regenerate RESULTS.md from the latest eval json
	$(PY) eval/report.py

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest tests/ -q

docker:
	docker build -t eu-ai-act-rag .
	docker run --rm -p 8501:8501 --env-file .env eu-ai-act-rag

all: corpus index eval-retrieval

clean:
	rm -rf data/index data/processed __pycache__ .pytest_cache
