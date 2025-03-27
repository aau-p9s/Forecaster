FROM unit8/darts:latest

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Api ./Api 
COPY Assets ./Assets
COPY Database ./Database 
COPY ML ./ML
COPY Utils ./Utils
COPY app.py .


RUN chmod +x /src/app.py

ENTRYPOINT ["/src/app.py"]
