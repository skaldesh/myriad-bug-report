# Issue
After some time the inference fails with the error message `Exception in inference routine: Failed to queue inference: NC_ERROR`.

# Docker environment
Using the official OpenVINO docker image `ubuntu20_dev:latest`  
The command that was used binds the myriad device to the container and runs the container in privileged mode:  
```
docker run \
    --name myrbug \
    -it \
    --rm \
    --net=host \
    --device-cgroup-rule='c 189:* rmw' \
    -v /dev:/dev \
    -v /<path-to-repo>/model.bin:/model.bin:ro \
    -v /<path-to-repo>/model.xml:/model.xml:ro \
    -v /<path-to-repo>:/data \
    --log-driver=journald \
    --log-opt mode=non-blocking \
    --log-opt max-buffer-size=4m \
    --log-opt tag="{{.Name}}" \
    --privileged \
    -u root:root \
    openvino/ubuntu20_dev:latest \
        /bin/bash
```

## Build & run sample
Execute the following commands inside the container:  
- apt-get -y update && apt-get -y install build-essential
- cd /data
- cmake .
- make
- ./sample

You can detach from the container with `Ctrl+p` followed by `Ctrl+q`  

## Example error logs
See the files `errorlog1.txt` and `errorlog2.txt` for sample output.  
The logs stem from different systems with the same configuration

## Model
The model is a [pretrained model from the OpenVINO model zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-detection-0200)