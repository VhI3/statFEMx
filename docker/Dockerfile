FROM dolfinx/dolfinx:stable

WORKDIR /opt/statfemx

COPY pyproject.toml README.md /opt/statfemx/
COPY data /opt/statfemx/data
COPY src /opt/statfemx/src
COPY scripts /opt/statfemx/scripts
COPY notebooks /opt/statfemx/notebooks
COPY tests /opt/statfemx/tests

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -e ".[dev,notebooks]"

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MPLBACKEND=Agg

EXPOSE 8888

CMD ["python3", "scripts/run_bar1d_full_pipeline.py", "--backend", "fenicsx", "--output", "bar1d_results.npz"]
