FROM unit8/darts:latest

WORKDIR /src

COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /src/app.py

ENTRYPOINT ["/src/app.py"]
