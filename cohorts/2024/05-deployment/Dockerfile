FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv

RUN pipenv install --system --deploy

COPY ["app.py", "./"]

COPY ["homework", "./homework"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "app:app"]
