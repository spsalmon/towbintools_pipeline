#!/bin/bash

rm conda-lock.yaml

~/.local/bin/micromamba run -n towbintools conda-lock -f environment.yaml -p linux-64 --lockfile conda-lock.yaml
