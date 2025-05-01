{ pkgs, config, lib, ... }:
with pkgs;
with config;

let

    compose_file = writeText "docker-compose.json" (lib.strings.toJSON {
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
        };
    });

in

writeScriptBin "reinitialize" ''
    #!${bash}/bin/bash
    set -e

    printf "\U0001F40B starting db\n"
    ${docker}/bin/docker compose -f ${compose_file} up -d --remove-orphans 2> ./logs/docker-compose.txt

    echo "sleeping to let postgres start"
    sleep 5

    export PGPASSWORD=${postgres_password}
    echo "dropping db"
    ${postgresql}/bin/dropdb -h localhost -U ${postgres_user} ${postgres_database} --no-password
    echo "initializing db"
    ${postgresql}/bin/createdb -h localhost -U ${postgres_user} ${postgres_database} --no-password

    rm -fr /tmp/autoscaler
    ${pkgs.git}/bin/git clone https://github.com/aau-p9s/Autoscaler /tmp/autoscaler
    echo "migrating db"
    ${dotnet-sdk_8}/bin/dotnet run --project /tmp/autoscaler/Autoscaler.DbUp > ./logs/dbup.txt

    printf "\U0001F40B stopping db\n"
    ${docker}/bin/docker compose -f ${compose_file} down 2> ./logs/docker-compose.txt
''
