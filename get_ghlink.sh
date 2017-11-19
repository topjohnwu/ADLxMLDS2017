#!/usr/bin/env bash
TOKEN=8f721a61667c6b2052b76170bca4b0755cd45ad3
curl \
	-H "Authorization: token $TOKEN" \
	-H "Accept: application/vnd.github.v3.raw" \
	-s "https://api.github.com/repos/topjohnwu/ADLxMLDS2017/releases"
