{ pkgs, config, ... }:

with pkgs;

writeScriptBin "dump" ''
    #!${bash}/bin/bash
    export PGPASSWORD=${postgres_password}
    ${postgresql}/bin/pg_dump -U ${config.postgres_user} -h ${config.postgres_address} ${config.postgres_database} > ./database.sql
''
