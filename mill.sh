#!/bin/bash

[ -z "$4" ] || tt=".$4"


if [ "$1" = "run" ]; then
    papermill -f $2/$3$tt.yml $2/$3.ipynb temp/notebooks/$3.ipynb --cwd $(pwd)/$2
else
    papermill -f $2/$3$tt.yml --prepare-only $2/$3.ipynb $2/$3.ipynb
fi

