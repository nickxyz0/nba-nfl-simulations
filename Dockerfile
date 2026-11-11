# Hosted copy of the combined site (Hugging Face Spaces, Docker SDK). Local use doesn't need this:
# run `streamlit run app.py` from the repository root instead.
FROM python:3.9-slim

# xgboost needs the OpenMP runtime, which the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000; the app writes its downloaded data next to the code.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 SIM_HOSTED=1
WORKDIR /home/user/app

COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir --user -r requirements-web.txt
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
