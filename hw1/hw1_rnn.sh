#!/usr/bin/env bash
if [ ! -d models ]; then
	mkdir models
	curl -L -H \
	'Accept: application/octet-stream' \
	'https://8f721a61667c6b2052b76170bca4b0755cd45ad3:@api.github.com/repos/topjohnwu/ADLxMLDS2017/releases/assets/5187687' \
	| tar zxf - -C models
fi
python run.py test rnn 500 "$1" rnn.ckpt "$2"
