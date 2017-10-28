#!/usr/bin/env bash
if [ ! -d models ]; then
	mkdir models
	curl -L https://www.dropbox.com/s/1ojzxc41dmiinhk/models.tgz?dl=1 | tar zxf - -C models
fi
python run.py test rnn 500 "$1" rnn.ckpt "$2"
