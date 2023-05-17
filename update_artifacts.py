# !/usr/bin/env python3

import subprocess

with open(".artifactinclude", "r") as f:
    artifacts = f.readlines()

# write rsync file
with open(".artifactinclude_rsync", "w") as f:
    for artifact in artifacts:
        # split path
        path = artifact.split("/")
        for i in range(len(path)):
            # write path
            f.write("+ " + "/".join(path[: i + 1]) + "\n")
    f.write("- *")


subprocess.call(
    [
        "rsync",
        "-av",
        "--include-from=.artifactinclude_rsync",
        "data/",
        "artifacts/",
        # "--delete-excluded",
    ]
)
