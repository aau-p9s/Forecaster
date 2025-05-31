{ pkgs, config, ... }:
with pkgs;
with config;

writeText "docker-compose.json" (lib.strings.toJSON {
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
        hostname = postgres_address;
        networks = [network];
    };
    networks.${network} = {};
})
