#!/usr/bin/env bash
if [ ! -d models ]; then
	mkdir models
	curl -L https://www.dropbox.com/s/1ojzxc41dmiinhk/models.tgz?dl=1 | tar zxf - -C models
fi
python run.py test cnn 100 "$1" cnn.ckpt "$2"
