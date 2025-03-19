FROM python:3.12-slim

WORKDIR /src

COPY . /src

RUN apt update && apt install -y libgbm-dev libgomp1

RUN pip install -r requirements.txt

RUN chmod +x /src/app.py

ENTRYPOINT ["/src/app.py"]
