FROM python:3.12-slim

WORKDIR /src

COPY . /src

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]
