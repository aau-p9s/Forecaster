FROM unit8/darts:0.32.0

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY Api ./Api 
#COPY Assets/models ./Assets/models
COPY Assets/test_data.csv ./Assets
COPY Assets/test_model.pth ./Assets
COPY Database ./Database 
COPY ML ./ML
COPY Utils ./Utils
COPY app.py .


RUN chmod +x /src/app.py

ENTRYPOINT ["/src/app.py"]
