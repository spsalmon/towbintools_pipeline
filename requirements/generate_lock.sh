#!/bin/bash

rm conda-lock.yml

~/.local/bin/micromamba run -n towbintools conda-lock -f environment.yml -p linux-64 -p win-64 --kind lock
~/.local/bin/micromamba run -n towbintools conda-lock render --kind env
