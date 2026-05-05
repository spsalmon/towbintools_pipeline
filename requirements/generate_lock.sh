#!/bin/bash

rm *lock.yml

~/.local/bin/micromamba run -n towbintools conda-lock -f environment.yml -p linux-64 --kind lock
~/.local/bin/micromamba run -n towbintools conda-lock render --kind env
