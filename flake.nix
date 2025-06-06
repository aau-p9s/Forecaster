{
    inputs.nixpkgs.url = "nixpkgs/nixos-24.05";
    inputs.ml-pkgs.url = "github:nixvital/ml-pkgs/archive/24.05";
    inputs.ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    outputs = inputs: let
        system = "x86_64-linux";
        pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
                inputs.ml-pkgs.overlays.time-series
            ];
        };
    in {
        devShells.${system}.default = pkgs.mkShellNoCC {

            packages = with pkgs; [
                inputs.self.packages.${system}.env

                postgresql
                (writeScriptBin "p10psql" ''
                    #!${bash}/bin/bash
                    export PGPASSWORD=password
                    ${postgresql}/bin/psql -h localhost -p 5432 -U root autoscaler $@
                '')
                (writeScriptBin "p10dropdb" ''
                    #!${bash}/bin/bash
                    export PGPASSWORD=password
                    ${postgresql}/bin/dropdb -h localhost -p 5432 -U root autoscaler $@
                '')
                (writeScriptBin "p10createdb" ''
                    #!${bash}/bin/bash
                    export PGPASSWORD=password
                    ${postgresql}/bin/createdb -h localhost -p 5432 -U root autoscaler $@
                '')
            ];
        };

        packages.${system}.env = pkgs.python3.withPackages (py: with py; [
            psycopg2
            ipython
            cloudpickle
            flask
            flask-restx
            optuna
            # unstable shit
            darts
        ]);
    };
}
