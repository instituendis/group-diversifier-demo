# WARNING: the app has NO authentication. Publish the port on 127.0.0.1 only
# (`-p 127.0.0.1:8501:8501`, as compose.yaml does), never on 0.0.0.0.
FROM python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50

LABEL org.opencontainers.image.title="Group Diversifier" \
      org.opencontainers.image.description="Allocate students into diverse project teams (Streamlit app and CLI)" \
      org.opencontainers.image.version="0.0.4" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_BROWSER_SERVER_ADDRESS=localhost

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin gd
WORKDIR /app

COPY requirements.lock.txt ./
RUN pip install --require-hashes --no-deps -r requirements.lock.txt

COPY --chown=gd:gd app/ app/
COPY --chown=gd:gd config/ config/
COPY --chown=gd:gd data/ data/
COPY --chown=gd:gd .streamlit/ .streamlit/
# INVARIANT: DEMO_MOD[E] is a glob on purpose. The optional demo-mode marker file
# is copied when the repository has it, and nothing is copied (no error) when it
# does not. A plain name would fail the build wherever the file is absent.
COPY --chown=gd:gd LICENSE NOTICE THIRD_PARTY_LICENSES.md DEMO_MOD[E] ./

USER gd
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
