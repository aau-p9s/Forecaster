{ pkgs, config, lib, ... }:
with pkgs;
with config;

writeScriptBin "reinit" ''
    #!${bash}/bin/bash
    set -e

    export PGPASSWORD=${postgres_password}
    echo "dropping db"
    ${postgresql}/bin/dropdb -h ${postgres_address} -U ${postgres_user} ${postgres_database} --no-password
    echo "initializing db"
    ${postgresql}/bin/createdb -h ${postgres_address} -U ${postgres_user} ${postgres_database} --no-password

    rm -fr /tmp/autoscaler
    ${pkgs.git}/bin/git clone https://github.com/aau-p9s/Autoscaler /tmp/autoscaler
    echo "migrating db"
    ${dotnet-sdk_8}/bin/dotnet run --project /tmp/autoscaler/Autoscaler.DbUp > /tmp/dbup.stdout
''
