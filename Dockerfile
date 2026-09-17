# ASSRS API container for Cloud Run.
#
# THIS FILE LIVES AT THE REPOSITORY ROOT ON PURPOSE.
#
# Cloud Run's "continuously deploy from a repository" flow builds with the
# context set to the DIRECTORY CONTAINING THE DOCKERFILE — not the repo root,
# whatever path you type. With this file under api/, `COPY requirements.txt`
# silently picked up api/requirements.txt and the next line failed with
# "stat api/requirements.txt: file does not exist". Keeping it at the root
# makes the context the root, which is what the COPY lines below need: the API
# imports analysis_engine.py, data_manager.py, box_detection.py and friends
# straight from the root so the Streamlit app and the API always run the same
# analysis code.
#
# Cloud Build now goes through /cloudbuild.yaml, whose context is the directory
# holding it — the root — so the same COPY lines work there and the build also
# deploys. Locally: docker build -t assrs-api .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # One BLAS thread: Cloud Run gives this container 1 vCPU, and the HMM's
    # numpy/sklearn work oversubscribes and slows down with more.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Dependencies first, in their own layer, so a code-only change reuses it.
COPY requirements.txt /tmp/requirements-root.txt
COPY api/requirements.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-root.txt -r /tmp/requirements-api.txt

# Root-level Python modules (the shared analysis code) and the packages.
#
# `COPY *.py` takes FILES ONLY — a package directory needs its own COPY line or
# it is silently absent from the image, uvicorn cannot import the app, nothing
# binds to $PORT, and Cloud Run reports it as "failed to start and listen on
# the port" with no mention of the import. Every top-level package below must
# have a line here; api/tests/test_dockerfile.py fails the build if one does not.
COPY *.py ./
COPY api ./api
COPY markets ./markets
COPY strategies ./strategies

# Cloud Run injects PORT; 8080 is its default.
ENV PORT=8080
EXPOSE 8080

# One worker per container: Cloud Run scales by adding containers, and a
# second worker would just halve the memory each HMM run has to work with.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65
