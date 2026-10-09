PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup corpus index eval eval-retrieval eval-versions eval-paragraph eval-extended check-qa agreement app test report report-check docker clean all

setup:
	python3 -m venv .venv && $(PIP) install -q -U pip && $(PIP) install -q -r requirements.txt

corpus:          ## download + chunk the AI Act
	$(PY) -m src.euactrag.fetch
	$(PY) -m src.euactrag.ingest

index:           ## embed + build the vector store
	$(PY) -m src.euactrag.index

eval:            ## published evaluation, original text (needs GROQ_API_KEY)
	$(PY) eval/run_eval.py --corpus 2024-07-12

eval-retrieval:  ## retrieval metrics only, no API key needed
	$(PY) eval/run_eval.py --corpus 2024-07-12 --no-generation --tag retrieval_only

eval-versions:   ## retrieval on both texts of the Act, amended questions too, no API key
	$(PY) eval/run_eval.py --corpus 2024-07-12 --qa amended --no-generation --tag amended_2024-07-12
	$(PY) eval/run_eval.py --corpus 2026-07-27 --qa amended --no-generation --tag amended_2026-07-27
	$(PY) eval/run_eval.py --corpus 2026-07-27 --qa original --no-generation --tag original_2026-07-27

eval-paragraph:  ## article against paragraph chunks, paragraph-level citation metrics, no API key
	for g in article paragraph; do \
	  $(PY) eval/run_eval.py --corpus 2024-07-12 --qa original --granularity $$g --no-generation --tag cite_$${g}_original_2024-07-12 && \
	  $(PY) eval/run_eval.py --corpus 2026-07-27 --qa amended --granularity $$g --no-generation --tag cite_$${g}_amended_2026-07-27 || exit 1; \
	done

eval-extended:   ## all 122 questions current for the 2026 text, both chunkings, no API key
	for g in article paragraph; do \
	  $(PY) eval/run_eval.py --corpus 2026-07-27 --qa extended --granularity $$g --no-generation --tag $${g}_extended_2026-07-27 && \
	  $(PY) eval/run_eval.py --corpus 2026-07-27 --qa all --granularity $$g --no-generation --tag $${g}_all_2026-07-27 || exit 1; \
	done

check-qa:        ## every gold reference and quote checked against the text
	$(PY) eval/check_qa.py

agreement:       ## judge and rule-based metrics against my hand labels, no API key
	$(PY) eval/agreement.py

report:          ## regenerate RESULTS.md from the latest eval json
	$(PY) eval/report.py

report-check:    ## fail if RESULTS.md no longer matches what report.py generates
	$(PY) eval/report.py --check

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
