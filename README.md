# Forecaster
This portion of this project is focused on machine learning.

# Prerequisites
This project was built with kubernetes in mind, therefore it heavily uses github actions to build and tag images that are pullable by kubernetes clusters. This means that this code is not meant to be run outside of kubernetes, and containerized environments.

* To use the pytorch models, you **HAVE** to have an Nvidia Accelerator capable of training in a reasonable time.
* Kubernetes or Docker Compose

# Build
If you have another registry, you could build it yourself and push it manually, to do so you would have to follow these steps:

```sh
docker build . -t <your-tag>
docker push <your-tag>

```

# Usage
I would recommend to use our ready to deploy setup at: <a href=https://skade.dev/api/p10/k8s/autoscaler.yml>Here</a>

The manual setup would require looking into our environment variable setup below

# Environment variables
The following variables are provided:
| Name                             | default    | description                                                                          |
| ----                             | -------    | -----------                                                                          |
| FORECASTER__PGSQL__DATABASE      | autoscaler | Postgresql database name                                                             |
| FORECASTER__PGSQL__USER          | root       | Postgresql user name                                                                 |
| FORECASTER__PGSQL__PASSWORD      | password   | Postgresql password                                                                  |
| FORECASTER__PGSQL__ADDR          | 0.0.0.0    | Postgresql address                                                                   |
| FORECASTER__PGSQL__PORT          | 5432       | Postgresql port                                                                      |
| FORECASTER__ADDR                 | 0.0.0.0    | Forecaster address                                                                   |
| FORECASTER__PORT                 | 8080       | Forecaster port                                                                      |
| FORECASTER__TRAIN_TIMEOUT        | -1         | Timeout for model training                                                           |
| FORECASTER__TRAIN__GPUS          | 2          | Number of GPU's for training                                                         |
| FORECASTER__CLEAR__DATA          | false      | Whether the Forecaster should clear the historic and forecasts table before starting |
| FORECASTER__TEMPORARY__DIRECTORY | /dev/shm   | Temporary directory to store intermediate files during model serialization           |
| FORECASTER__ENABLE__GPU          | true       | Whether to use GPU Acceleration                                                      |
