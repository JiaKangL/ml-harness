#!/usr/bin/env bash
# Fetch KuaiRand-Pure (47 MB download, ~200 MB extracted) into the starter kit.
# Zenodo direct link, no registration required.
set -euo pipefail
cd "$(dirname "$0")/../kuairand-starter-kit"
URL="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
if [ -d KuaiRand-Pure/data ]; then echo "already present"; exit 0; fi
curl -fL -o KuaiRand-Pure.tar.gz "$URL"
tar xzf KuaiRand-Pure.tar.gz
rm -f KuaiRand-Pure.tar.gz
echo "-> $(pwd)/KuaiRand-Pure/data"
