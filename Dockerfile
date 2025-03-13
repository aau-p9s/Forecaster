FROM python:3.12

WORKDIR /src

COPY . /src

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]
