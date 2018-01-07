#! /usr/bin/env bash
if [ ! -d models ]; then
	curl -L 'https://www.dropbox.com/s/ece3k3ewn16jbvg/models.tar.gz?dl=1' | tar zxvf -
fi

python3.5 run.py --test "$1"
