#!/usr/bin/python

import sys
import subprocess

msg = "expecting arguments to be infile and outfile"
assert len(sys.argv) > 2
args = {"infile": sys.argv[1], "outfile": sys.argv[2]}

cmd = """
ffmpeg -i {infile}.ogv \
       -c:v libx264 -preset veryslow -crf 22 \
       -c:a aac -b:a 160k -strict -2 -b:v 1M\
        {outfile}.mp4
""".format(**args)

subprocess.check_output(cmd.split())
