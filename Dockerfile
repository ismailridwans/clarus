FROM python:3.11-slim

# OPENAI_API_KEY is a RUNTIME variable — inject it via:
#   docker run -e OPENAI_API_KEY=sk-... clarus
# or via HuggingFace Space Secrets in the Space settings UI.
# Do NOT bake it into the image.

WORKDIR /app

# Install Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Download CMS reference data (falls back to committed bundle files if offline)
RUN python data/download.py

# Verify reference DB loads correctly — build fails if data is broken
RUN python -c "\
from data.setup import load_all; import sqlite3; \
from server.schema import create_tables; \
db = sqlite3.connect(':memory:'); create_tables(db); load_all(db); \
cpt  = db.execute('SELECT COUNT(*) FROM cpt_codes').fetchone()[0]; \
ncci = db.execute('SELECT COUNT(*) FROM ncci_edits').fetchone()[0]; \
plans= db.execute('SELECT COUNT(*) FROM plan_templates').fetchone()[0]; \
assert cpt > 0 and ncci > 0 and plans == 8, \
    f'Data missing: {cpt} CPT, {ncci} NCCI, {plans} plans'; \
print(f'Ref DB OK: {cpt} CPT codes, {ncci} NCCI edits, {plans} plans') \
"

# Run environment self-tests — build fails if the environment is broken.
# These are deterministic (no LLM calls, no OPENAI_API_KEY needed).
RUN python -m pytest tests/test_grader.py tests/test_episodes.py -q --tb=short

EXPOSE 7860

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
