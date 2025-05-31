{ pkgs, config, ... }:

with pkgs;

writeScriptBin "start-database" ''
    #!${bash}/bin/bash

    printf "\U0001F40B starting db\n"
    ${docker}/bin/docker compose -f ${callPackage ./docker-compose.nix { inherit config; }} up -d --remove-orphans
''
