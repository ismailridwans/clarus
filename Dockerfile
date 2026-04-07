FROM python:3.11-slim

# OPENAI_API_KEY is a RUNTIME variable — inject it via:
#   docker run -e OPENAI_API_KEY=sk-... clarus
# or via HuggingFace Space → Settings → Variables and secrets.
# Do NOT bake it into the image.

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Set PYTHONPATH so all modules resolve from /app
ENV PYTHONPATH=/app

# Download CMS reference data (falls back to committed bundles if offline)
RUN python data/download.py

# Verify reference DB loads and bundle data is intact
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

# Smoke test — build fails if the environment is broken.
# Uses asyncio directly (no pytest needed at build time).
RUN python -m tests.self_test

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
