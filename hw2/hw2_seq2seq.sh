#!/usr/bin/env bash
if [ ! -d models/vanilla ]; then
	mkdir models
	curl -L -H \
	'Accept: application/octet-stream' \
	'https://8f721a61667c6b2052b76170bca4b0755cd45ad3:@api.github.com/repos/topjohnwu/ADLxMLDS2017/releases/assets/5390031' \
	| tar zxf - -C models
fi
python model_seq2seq.py test vanilla "$1" "$2"
python model_seq2seq.py peer vanilla "$1" "$3"
