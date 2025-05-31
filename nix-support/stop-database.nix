{ pkgs, config, ... }:

with pkgs;

writeScriptBin "stop-database" ''
    #!${bash}/bin/bash

    printf "\U0001F40B stopping db\n"
    ${docker}/bin/docker compose -f ${callPackage ./docker-compose.nix { inherit config; }} down --remove-orphans
''
