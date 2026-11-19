#!/usr/bin/env python3
"""same_sass.py BASE NEW: is every GPU kernel of BASE in NEW, instruction for instruction?

Dumps the SASS of both binaries (cuobjdump -sass), keys each kernel by its name without template arguments or
parameters (head_norm_rope_kernel<false>(...) -> head_norm_rope_kernel), and looks for each of BASE's kernel bodies
among NEW's bodies of the same name. A kernel body is its instruction lines with their encodings (addresses are
relative to the kernel), so "found" means byte-identical machine code. Prints the count, the kernels that are not
found (with the first lines that differ), and the kernel names that are new. Exit 0 when every BASE kernel is found.
"""
import collections
import difflib
import re
import shutil
import subprocess
import sys

CUOBJDUMP = shutil.which("cuobjdump") or "/usr/local/cuda-13.4/bin/cuobjdump"


def kernels(binary):
    """-> {short name: [body, ...]}, {short name: set of demangled names}"""
    sass = subprocess.run([CUOBJDUMP, "-sass", binary], capture_output=True, text=True, check=True).stdout
    bodies, names, mangled, cur = [], [], [], None
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = []
            bodies.append(cur)
            mangled.append(m.group(1))
        elif cur is not None and line.strip() and not line.strip().startswith("....."):
            cur.append(line.strip())
    demangled = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True,
                               check=True).stdout.splitlines()
    by_name, full = collections.defaultdict(list), collections.defaultdict(set)
    for d, body in zip(demangled, bodies):
        short = re.sub(r"^void ", "", d)
        short = re.split(r"[<(]", short, maxsplit=1)[0]
        by_name[short].append("\n".join(body))
        full[short].add(d)
    return by_name, full


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base, base_full = kernels(sys.argv[1])
    new, new_full = kernels(sys.argv[2])
    total = sum(len(v) for v in base.values())
    missing = []
    for name, bodies in base.items():
        pool = collections.Counter(new.get(name, []))
        for body in bodies:
            if pool[body] > 0:
                pool[body] -= 1
            else:
                missing.append((name, body))
    print(f"{total - len(missing)} of {total} kernels of {sys.argv[1]} are in {sys.argv[2]} with the same machine code")
    for name, body in missing:
        print(f"  NOT FOUND: {name} ({', '.join(sorted(base_full[name]))[:200]})")
        others = new.get(name, [])
        if others:
            strip = lambda b: [re.sub(r"/\*[^*]*\*/", "", x).strip() for x in b.splitlines()]
            near = max(others, key=lambda o: difflib.SequenceMatcher(None, body, o).ratio())
            diff = list(difflib.unified_diff(strip(body), strip(near), "base", "new", n=1, lineterm=""))
            print("    " + "\n    ".join(diff[:24]))
    added = sorted(d for name in new_full for d in new_full[name] - base_full.get(name, set()))
    print(f"kernels in {sys.argv[2]} by names {sys.argv[1]} does not have: {len(added)}")
    for d in added:
        print("  + " + d[:160])
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
