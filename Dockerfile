FROM python:3.13

WORKDIR /src

COPY . /src

RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

RUN chmod +x /src/app.py

ENTRYPOINT ["/src/app.py"]
