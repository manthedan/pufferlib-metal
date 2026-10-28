#!/usr/bin/env python3
"""Public Protein Sobol32: Joe--Kuo D(6), with a fixed digital XOR shift.

Source: https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
Copyright (c) 2008 Frances Y. Kuo and Stephen Joe; BSD notice in
tools/metal/joe-kuo/LICENSE and NOTICE.

The compact artifact covers exactly Protein's supported 20,000 dimensions.
Dimension 1 is implicit. For each remaining row: one degree byte s, then
ceil(((s-1) + s*(s-1)/2)/8) little-endian packed bytes: a in s-1 bits,
followed by m_j >> 1 in j-1 bits (j=1..s). The omitted low bits are all 1.
Unused high bits are zero. No NVIDIA data is used.

Expanded rows are 32 little-endian uint32 directions then one XOR constant.
Constants are the LOW 32 bits of successive SplitMix64 outputs, starting at
PROTEIN_SOBOL_SHIFT_SEED = 0x50554646534f424c (ASCII "PUFFSOBL"). Each draw
adds the golden-ratio increment before mixing; arithmetic wraps at 64 bits.
This is a digital-shift scramble, not cuRAND's scrambled-direction transform.
"""
import argparse
import hashlib
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = 20000
SEED = 0x50554646534F424C
COMPACT_SHA256 = 'adca16e3a020a0ee099fd130b6cb5f46628c14fe629909b7fa850eb585d07210'
FULL_SOURCE_SHA256 = '68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441'
EXCERPT_SHA256 = 'af7492ff6eb0c4e1d968a15710d6496e61709d151821859841b0c16893d7b905'
URLS = (
    'https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201',
    'https://raw.githubusercontent.com/joe-kuo/sobol_data/master/new-joe-kuo-6.21201',
)
ARTIFACT = ROOT / 'src/metal/protein_sobol_public.bin'
EXCERPT = ROOT / 'tools/metal/joe-kuo/new-joe-kuo-6.20000'


def checked_excerpt(data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXCERPT_SHA256:
        raise ValueError(f'Joe-Kuo excerpt SHA256 mismatch: {digest}; expected {EXCERPT_SHA256}')
    return data


def excerpt_from_full(data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != FULL_SOURCE_SHA256:
        raise ValueError(
            f'full Joe-Kuo source SHA256 mismatch: {digest}; expected {FULL_SOURCE_SHA256}')
    lines = data.splitlines(keepends=True)
    if len(lines) < DIMENSIONS:
        raise ValueError('incomplete full Joe-Kuo source')
    return checked_excerpt(b''.join(lines[:DIMENSIONS]))


def source_rows(data):
    data = checked_excerpt(data)
    rows = []
    for dimension, line in enumerate(data.decode('ascii').splitlines()[1:DIMENSIONS], 2):
        d, s, a, *m = map(int, line.split())
        if d != dimension or not 1 <= s <= 18 or len(m) != s or not 0 <= a < 1 << (s-1):
            raise ValueError(f'invalid Joe-Kuo row {dimension}')
        if any(v % 2 != 1 or not 0 < v < 1 << j for j, v in enumerate(m, 1)):
            raise ValueError(f'invalid initial direction in row {dimension}')
        rows.append((s, a, m))
    if len(rows) != DIMENSIONS - 1:
        raise ValueError('incomplete Joe-Kuo source')
    return rows


def pack_rows(rows):
    result = bytearray()
    for s, a, m in rows:
        bits, count = a, s - 1
        for j, v in enumerate(m, 1):
            bits |= (v >> 1) << count
            count += j - 1
        result.append(s)
        result.extend(bits.to_bytes((count + 7) // 8, 'little'))
    return bytes(result)


def unpack_rows(data):
    rows, offset = [], 0
    for _ in range(DIMENSIONS - 1):
        s = data[offset]
        if not 1 <= s <= 18:
            raise ValueError('invalid compact degree')
        size = ((s-1) + s*(s-1)//2 + 7)//8
        bits = int.from_bytes(data[offset+1:offset+1+size], 'little')
        a, bits = bits & ((1 << (s-1))-1), bits >> (s-1)
        m = []
        for j in range(1, s+1):
            m.append(((bits & ((1 << (j-1))-1)) << 1) | 1)
            bits >>= j-1
        if bits:
            raise ValueError('nonzero compact padding')
        rows.append((s, a, m))
        offset += 1 + size
    if offset != len(data):
        raise ValueError('incorrect compact size')
    return rows


def splitmix64(state):
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return state, z ^ (z >> 31)


def expand_rows(rows):
    directions = [[1 << (31-b) for b in range(32)]]
    for s, a, m in rows:
        v = [x << (31-b) for b, x in enumerate(m)]
        for b in range(s, 32):
            x = v[b-s] ^ (v[b-s] >> s)
            for k in range(1, s):
                if a & (1 << (s-1-k)):
                    x ^= v[b-k]
            v.append(x)
        directions.append(v)
    state = SEED
    for v in directions:
        state, shift = splitmix64(state)
        v.append(shift & 0xffffffff)
    return b''.join(struct.pack('<33I', *v) for v in directions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--source', type=Path,
                        help='full original Joe-Kuo text; verify SHA256 and committed excerpt')
    source.add_argument('--download', action='store_true',
                        help='download the full source, then verify SHA256 and committed excerpt')
    source.add_argument('--excerpt', type=Path,
                        help='pinned 20,000-dimension source excerpt (default: committed excerpt)')
    source.add_argument('--from-compact', type=Path, help='expand a pinned compact artifact offline')
    parser.add_argument('--output', type=Path, default=ARTIFACT)
    parser.add_argument('--expanded-output', type=Path)
    parser.add_argument('--check', action='store_true', help='compare output byte-for-byte, do not write')
    args = parser.parse_args()
    if args.from_compact:
        compact = args.from_compact.read_bytes()
        if hashlib.sha256(compact).hexdigest() != COMPACT_SHA256:
            raise ValueError('compact artifact SHA256 mismatch')
        rows = unpack_rows(compact)
    else:
        if args.download:
            for url in URLS:
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        data = response.read()
                    break
                except OSError:
                    if url == URLS[-1]:
                        raise
            excerpt = excerpt_from_full(data)
            committed = checked_excerpt(EXCERPT.read_bytes())
            if excerpt != committed:
                raise ValueError('full Joe-Kuo source excerpt differs from committed excerpt')
        elif args.source:
            excerpt = excerpt_from_full(args.source.read_bytes())
            committed = checked_excerpt(EXCERPT.read_bytes())
            if excerpt != committed:
                raise ValueError('full Joe-Kuo source excerpt differs from committed excerpt')
        else:
            excerpt = (args.excerpt or EXCERPT).read_bytes()
        rows = source_rows(excerpt)
    compact = pack_rows(rows)
    if args.check:
        if args.output.read_bytes() != compact:
            raise ValueError(f'generated table differs: {args.output}')
    else:
        args.output.write_bytes(compact)
    print(f'{len(compact)} bytes SHA256 {hashlib.sha256(compact).hexdigest()} {args.output}')
    if args.expanded_output:
        expanded = expand_rows(rows)
        args.expanded_output.write_bytes(expanded)
        print(f'{len(expanded)} bytes SHA256 {hashlib.sha256(expanded).hexdigest()} expanded')


if __name__ == '__main__':
    main()
