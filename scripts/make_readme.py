#!/usr/bin/env python3
# AUTHOR: rendaw (original), alexlyttle (modified)
# LICENSE: https://opensource.org/licenses/BSD-2-Clause
import glob
import re
import os.path

from datetime import datetime, timezone

for source in glob.glob('./**/README.rst.src', recursive=True):
    dirname = os.path.dirname(source)

    def include(match):
        with open(os.path.join(dirname, match.group('filename')), 'r') as f:
            body = f.read()

        return body

    dest = re.sub('\\.src', '', source)
    with open(source, 'r') as f:
        text = f.read()

    d = datetime.now(tz=timezone.utc)
    header = f".. {dest} file, created by\n" + \
             f"   {__file__} on {d.strftime('%c %Z')}.\n" + \
              "   ================ DO NOT MODIFY THIS FILE! =================\n" + \
              "   It is generated automatically as a part of a GitHub Action.\n" + \
              "   Any changes should be made to\n" + \
             f"   {dest}.src instead.\n" + \
              "   ===========================================================\n\n"

    text = re.sub(
        '^\\.\\. include:: (?P<filename>.*)$',
        include,
        text,
        flags=re.M,
    )
    with open(dest, 'w') as f:
        f.write(header + text)
