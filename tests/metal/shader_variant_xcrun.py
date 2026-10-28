#!/usr/bin/env python3
"""Test-only toolchain shim: build every shader with IEEE sigmoid division.

The ordinary build/link recipe and library embedding remain unchanged. The
real xcrun is resolved before this shim is put on PATH by the seed gate.
"""
import os
import sys

args = sys.argv[1:]
if 'metal' in args and '-c' in args:
    args.append('-DPUF_SIG_IEEE_DIV')
    with open(os.environ['PUFFER_VARIANT_BUILD_LOG'], 'a') as log:
        log.write(' '.join(args) + '\n')
os.execv(os.environ['PUFFER_REAL_XCRUN'], [os.environ['PUFFER_REAL_XCRUN']] + args)
