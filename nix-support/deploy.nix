{ pkgs, lib, config, ... }: 
with pkgs;
with config;

let 

    dockerfile = writeText "Dockerfile" ''
        FROM unit8/darts:latest

        WORKDIR /run 

        RUN pip install psycopg2_binary optuna cloudpickle

        COPY . .


        ENTRYPOINT [ "python", "./insert_cloudpickle.py" ]
    '';
    
in 

writeScriptBin "deploy" ''
    #!${bash}/bin/bash
    set -e

    printf "\U0001F40B building deployment container\n"
    ${docker}/bin/docker build . -f ${dockerfile} -t deploy:latest
    rm -f .main.py

    printf "\U0001F40B deploying db\n"
    ${docker}/bin/docker run deploy:latest --dbname ${postgres_database} --dbuser ${postgres_user} --dbhost ${postgres_address} --dbport ${postgres_port} --dbpassword ${postgres_password} 2> /tmp/deploy.stdout
''
