{ pkgs, lib, config, ... }: 
with pkgs;
with config;

let 

    initialize = writeScriptBin ".main.py" ''
        #!/usr/bin/env python

        from psycopg2 import connect
        from datetime import datetime
        from uuid import uuid4
        from Utils.initialize import main

        models = []
        ${builtins.concatStringsSep "\n" (map (model: ''
            try:
                from darts.models import ${model}
                models.append(${model})
            except:
                print("Warning! Failed to load ${model}")
        '') (builtins.attrNames (builtins.readDir ../Assets/models)))}

        connection = connect(database="${postgres_database}", user="${postgres_user}", password="${postgres_password}", host="${postgres_address}", port=${postgres_port})

        main(models, connection)
    '';
    dockerfile = writeText "Dockerfile" ''
        FROM unit8/darts:latest

        WORKDIR /run 

        RUN pip install psycopg2 optuna

        COPY . .
        COPY ./Assets/models ./models


        ENTRYPOINT [ "python", "./.main.py" ]
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
    ${docker}/bin/docker run --network store_${network} deploy:latest 2> /tmp/deploy.stdout
''
