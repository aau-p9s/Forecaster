{ pkgs, lib, config, ... }: 
with pkgs;
with config;

let 

    dockerfile = writeText "Dockerfile" ''
        FROM unit8/darts:latest

        WORKDIR /run 

        RUN pip install psycopg2 optuna cloudpickle

        COPY . .
        COPY ./Assets/models ./models


        ENTRYPOINT [ "python", "./insert_cloudpickle.py" ]
    '';
    
in 

writeScriptBin "deploy" ''
    #!${bash}/bin/bash
    set -e

    cat ${initialize}/bin/.main.py > .main.py
    printf "\U0001F40B building deployment container\n"
    ${docker}/bin/docker build . -f ${dockerfile} -t deploy:latest -q
    rm -f .main.py

    printf "\U0001F40B deploying db\n"
    ${docker}/bin/docker run deploy:latest 2> /tmp/deploy.stdout
''
