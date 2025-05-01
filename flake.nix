{
    inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
    outputs = inputs: let
        system = "x86_64-linux";
        pkgs = import inputs.nixpkgs { inherit system; };

        config = { 
            postgres_address = "postgres";
            postgres_port = "5432";
            postgres_password = "password";
            postgres_user = "root";
            postgres_database = "autoscaler";
            forecaster_address = "0.0.0.0";
            forecaster_port = "8081";
            network = "autoscaler";
            postgres_data_source = "/var/postgres_data";
        };

    in {
        packages.${system} = {
            libs = pkgs.callPackage ./libs {};
            default = pkgs.callPackage ./nix-support/deploy.nix { inherit config; };
            deploy = pkgs.callPackage ./nix-support/deploy.nix { inherit config; };
            reinit = pkgs.callPackage ./nix-support/reinit.nix { inherit config; };
            test-deploy = pkgs.callPackage ./nix-support/test-deploy.nix { inherit config; };
            start-database = pkgs.callPackage ./nix-support/start-database.nix { inherit config; };
            stop-database = pkgs.callPackage ./nix-support/stop-database.nix { inherit config; };
            show_config = pkgs.writeScriptBin "show-config" ''
                #!${pkgs.bash}/bin/bash
                set -e

                echo "Config:"
                echo '${pkgs.lib.strings.toJSON config}' | ${pkgs.jq}/bin/jq
            '';
        };
        devShells.${system}.default = pkgs.mkShellNoCC {
            packages = with pkgs; [
                postgresql
            ];
        };
    };
}
