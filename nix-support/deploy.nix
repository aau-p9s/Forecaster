{ pkgs, lib, config, ... }: 
with pkgs;
with config;

let 

    compose-file = writeText "docker-compose.json" (lib.strings.toJSON {
        services.postgres = {
            image = "postgres";
            environment = {
                POSTGRES_USER = postgres_user;
                POSTGRES_DB = postgres_database;
                POSTGRES_PASSWORD = postgres_password;
            };
            ports = [
                "${postgres_port}:${postgres_port}"
            ];
            volumes = [
                "${postgres_data_source}:/var/lib/postgresql/data"
            ];
            hostname = postgres_address;
            networks = [network];
        };
        services.forecaster = {
            image = "ghcr.io/aau-p9s/forecaster:latest";
            environment = {
                FORECASTER__PGSQL__DATABASE = postgres_database;
                FORECASTER__PGSQL__USER = postgres_user;
                FORECASTER__PGSQL__PASSWORD = postgres_password;
                FORECASTER__PGSQL__ADDR = postgres_address;
                FORECASTER__PGSQL__PORT = postgres_port;
                FORECASTER__ADDR = forecaster_address;
                FORECASTER__PORT = forecaster_port;
            };
            ports = [
                "${forecaster_port}:${forecaster_port}"
            ];
            networks = [network];
        };
        networks.${network} = {};
    });

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

    rm -rf logs
    mkdir -p ./logs

    printf "\U0001F40B starting db\n"
    ${docker}/bin/docker compose -f ${compose-file} up -d --remove-orphans 2> ./logs/docker-compose.txt

    cat ${initialize}/bin/.main.py > .main.py
    printf "\U0001F40B building deployment container\n"
    ${docker}/bin/docker build . -f ${dockerfile} -t deploy:latest 2> ./logs/build.txt
    rm -f .main.py

    printf "\U0001F40B deploying db\n"
    ${docker}/bin/docker run --network store_${network} deploy:latest 2> ./logs/deploy.txt

    printf "\U0001F40B stopping db\n"
    ${docker}/bin/docker compose -f ${compose-file} down 2> ./logs/docker-compose.txt

''
