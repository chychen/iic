#!/bin/bash
docker run --rm -dit --name jay_apache -p 12346:80 -v /home/dgx/jay/iic:/usr/local/apache2/htdocs/ -v /raid/InclusiveImagesChallenge:/usr/local/apache2/htdocs/inputs/ httpd
