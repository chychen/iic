#!/bin/bash
if [ -z "$1"];
then
	echo "GPU cannot set to the empty string, exit 0."
	exit 0
fi
echo "CUDA_VISIBLE_DEVICES=$1"
docker run --runtime=nvidia --rm -e CUDA_VISIBLE_DEVICES=$1 --network host --name=jay_py2 -it -p 127.0.0.1:12345:12345 -p 127.0.0.1:52525:52525 -v /home/jay/iic:/iic -w /iic  jaycase/iic:py2_v2 bash
