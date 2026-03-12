FROM dolfinx/dolfinx:stable

WORKDIR /opt/statfemx

COPY pyproject.toml README.md /opt/statfemx/
COPY src /opt/statfemx/src
COPY scripts /opt/statfemx/scripts

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir chaospy scipy numba pytest && \
    python3 -m pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

CMD ["python3", "scripts/run_bar1d_full_pipeline.py", "--backend", "fenicsx", "--output", "bar1d_results.npz"]
