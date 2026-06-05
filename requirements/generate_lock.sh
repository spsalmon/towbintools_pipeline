#!/bin/bash

rm *lock.yml

~/.local/bin/micromamba run -n towbintools conda-lock lock -f environment.yml -p linux-64 --micromamba
~/.local/bin/micromamba run -n towbintools conda-lock render
