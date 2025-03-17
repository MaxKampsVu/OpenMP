#!/usr/bin/env python3

import struct
import sys
import re
import glob
import argparse

from trace_lib import Trace

VERBOSE = False

# parse trace line and return (proc_id, trace_type, addr, size)
def parse_event(line):
    (p, iaddr, type_str, addr, size) = line.split()
    return (int(p[1:]), type_str, int(addr, 16), int(size, 16))

# Check if memory operand address is four byte aligned.
def alignment_ok(check_alignment, type_str, addr):
    if check_alignment and (type_str == 'R' or type_str == 'W'):
        if addr % 4 == 0:
            return True
        else:
            if VERBOSE:
                print(f'Filtered: {type_str} 0x{addr:x}', file=sys.stderr)
            return False
    return True

def main():
    parser = argparse.ArgumentParser(
            description='Convert traces to trf format')
    parser.add_argument('input_basename',
            help='Basename of the input traces')
    parser.add_argument('output_trace',
            help='The output trace in trf format')
    parser.add_argument('-d', '--disable_alignment_filter', action='store_true', default=False, help='Four byte alignment filtering is the default, this disables alignment filtering')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Show filtered events')
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    check_alignment = args.disable_alignment_filter == False

    print(f"Converting '{args.input_basename}*' traces to '{args.output_trace}...'", file=sys.stderr)

    # binary addres format:
    #addr 4b  4b  4b  4b
    #  0: p0, p1, p2, p3
    # 32: p0, p1, p2, p3
    # ...

    # open all thread trace files

    # traces = [[] for _ in range(nprocs)]
    infiles = []
    filtered = 0

    for f in sorted(glob.glob(f'{args.input_basename}*')):
        infiles.append(open(f, 'r'))

    active_traces = len(infiles)
    otrace = Trace(args.output_trace, active_traces)

    events_left = True
    while(events_left):
        events_left = False
        for (i, f) in enumerate(infiles):
            line = f.readline()
            if line:
                (p, type_str, addr, size) = parse_event(line)
                assert i == p
                if type_str != 'E': events_left = True

                if alignment_ok(check_alignment, type_str, addr):
                    otrace.entry_str_type(type_str, addr)
                else:
                    # Replace misaligned read/write with a nop
                    otrace.entry_str_type('N', 0x0)
                    filtered += 1
            else:
                otrace.nop()

    otrace.close_without_end()

    if check_alignment and filtered > 0:
        print(f'Filtered events: {filtered}', file=sys.stderr)

if __name__ == "__main__":
    main()

