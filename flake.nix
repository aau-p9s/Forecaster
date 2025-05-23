{
    inputs.nixpkgs.url = "nixpkgs/nixos-24.11";
    outputs = inputs@{ ... }: let
        system = "x86_64-linux";
        pkgs = import inputs.nixpkgs { inherit system; };
    in {
        devShells.${system}.default = pkgs.mkShellNoCC {
            packages = with pkgs; [
                (python312.withPackages (py: with py; [
                    psycopg2
                    ipython
                    cloudpickle
                ]))
                postgresql
            ];
        };
    };
}
