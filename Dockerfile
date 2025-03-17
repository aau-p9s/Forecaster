FROM python:3.12-slim

WORKDIR /src

COPY . /src

RUN pip install -r requirements.txt

RUN chmod +x /src/app.py

ENTRYPOINT ["python", "app.py"]
