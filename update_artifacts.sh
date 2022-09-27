#!/bin/bash

set -e

rsync -av --include-from=.artifactinclude data/ artifacts/ --delete-excluded
