FROM unit8/darts:latest

WORKDIR /src

COPY Api ./Api 
COPY Assets ./Assets
COPY Database ./Database 
COPY ML ./ML
COPY Utils ./Utils
COPY app.py .
COPY requirements.txt .
COPY test_data.csv .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /src/app.py

ENTRYPOINT ["/src/app.py"]
