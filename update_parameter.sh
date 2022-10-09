#!/bin/bash

[ -z "$2" ] || tt=".$2"

papermill -f $1$tt.yml --prepare-only $1.ipynb $1.ipynb
