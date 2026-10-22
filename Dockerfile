# =============================================================================
# Production Dockerfile for AI Content Moderation System (Streamlit + ML)
# Optimized for local Docker Desktop and public cloud hosting (Render Web Service)
# =============================================================================

# Official stable Python slim base image
FROM python:3.11-slim

# Set production environment variables:
# - PYTHONDONTWRITEBYTECODE: Prevents creation of transient .pyc files
# - PYTHONUNBUFFERED: Keeps stdout/stderr streams unbuffered for real-time logging
# - STREAMLIT_SERVER_HEADLESS: Ensures Streamlit runs headlessly without browser popups
# - STREAMLIT_BROWSER_GATHER_USAGE_STATS: Disables telemetry prompt
# - STREAMLIT_SERVER_ENABLE_CORS: Prevents cross-origin websocket drops on reverse proxies
# - STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION: Allows proxied requests from public domains
# - PORT: Default local fallback port (Render injects its own PORT at runtime)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    PORT=8501

# Set container working directory
WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies using pre-compiled binary wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Streamlit application, test suite, and ML pipeline code
COPY app.py train.py test_project.py ./

# Copy required trained model artifacts and evaluation benchmarks
COPY models/ ./models/

# Expose default Streamlit port (local) and Render web service port
EXPOSE 8501 10000

# Dynamic healthcheck: checks localhost at the active runtime port (${PORT:-8501})
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8501'); urllib.request.urlopen(f'http://localhost:{port}/_stcore/health')" || exit 1

# Launch Streamlit bound to 0.0.0.0 and listening on dynamic ${PORT:-8501}
CMD ["sh", "-c", "python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
