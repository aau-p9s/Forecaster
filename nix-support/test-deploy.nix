{ pkgs, config, ... }: 
with pkgs;
with config;

let

in

writeScriptBin "test-deploy" ''
    #!${bash}/bin/bash
    set -e
    ${callPackage ./start-database.nix { inherit config; }}/bin/start-database

    echo "waiting to let db initialize..."
    sleep 5

    ${callPackage ./reinit.nix { inherit config; }}/bin/reinit

    ${callPackage ./deploy.nix { inherit config; }}/bin/deploy

    ${callPackage ./stop-database.nix { inherit config; }}/bin/stop-database
''
