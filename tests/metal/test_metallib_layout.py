#!/usr/bin/env python3
"""Shader payload growth must not relocate host code or globals.

Link two native fixtures using the production embedding segment, then compare
all host sections. --legacy-mutation deliberately restores the old segment
and must fail. The payloads are linker fixtures and are never executed.
"""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def host_sections(path):
    data = path.read_bytes()
    header = struct.unpack_from('<IiiIIIII', data)
    if header[0] != 0xfeedfacf:
        raise RuntimeError('expected a native 64-bit Mach-O fixture')
    offset = 32
    sections = {}
    for _ in range(header[4]):
        command, size = struct.unpack_from('<II', data, offset)
        if command == 0x19:  # LC_SEGMENT_64
            segment = struct.unpack_from('<II16sQQQQiiII', data, offset)
            for i in range(segment[-2]):
                sec = struct.unpack_from('<16s16sQQIIIIIIII', data, offset + 72 + 80 * i)
                name = sec[0].split(b'\0')[0].decode()
                owner = sec[1].split(b'\0')[0].decode()
                if name == '__metallib' or owner not in ('__TEXT', '__DATA', '__DATA_CONST'):
                    continue
                zero_fill = (sec[8] & 0xff) in (1, 0xc, 0x12)
                contents = None if zero_fill else data[sec[4]:sec[4] + sec[3]]
                sections[owner, name] = sec[2], sec[3], contents
        offset += size
    return sections


def main():
    recipe = (ROOT / 'tools/metal/build_native.sh').read_text()
    match = re.search(r'-Wl,-sectcreate,([^,]+),__metallib,', recipe)
    if not match:
        raise RuntimeError('native embedding segment not found')
    segment = '__DATA' if '--legacy-mutation' in sys.argv else match[1]
    directory = ROOT / 'build/metal'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='layout_', dir=directory) as tmp:
        tmp = Path(tmp)
        source = tmp / 'fixture.cpp'
        source.write_text('volatile int initialized = 7;\nvolatile int zeroed;\n'
                          'int main() { return initialized + zeroed; }\n')
        obj = tmp / 'fixture.o'
        subprocess.run(['clang++', '-O2', '-c', str(source), '-o', str(obj)], check=True)
        images = []
        for size in (4096, 65536):
            payload = tmp / f'payload_{size}.bin'
            payload.write_bytes(bytes(size))
            binary = tmp / f'fixture_{size}'
            subprocess.run(['clang++', str(obj),
                            f'-Wl,-sectcreate,{segment},__metallib,{payload}',
                            '-o', str(binary)], check=True)
            images.append(host_sections(binary))
        required = {('__TEXT', '__text'), ('__DATA', '__data')}
        if not required.issubset(images[0]) or not any(
                name in ('__bss', '__common') for owner, name in images[0]):
            raise RuntimeError('fixture is missing observable host code or globals')
        changed = [f'{owner},{name}' for owner, name in images[0].keys() | images[1].keys()
                   if images[0].get((owner, name)) != images[1].get((owner, name))]
        if changed:
            print('FAIL: shader payload moved host sections: ' + ', '.join(changed))
            return 1
        print(f'PASS: {segment} payload growth 4 -> 64 KiB leaves host sections identical')
        return 0


if __name__ == '__main__':
    sys.exit(main())
