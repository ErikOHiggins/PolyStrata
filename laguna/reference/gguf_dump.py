import struct, sys, collections

TYPES = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32', 7: 'bool', 8: 'str', 9: 'arr', 10: 'u64', 11: 'i64', 12: 'f64'}
FMT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
GGML = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
        13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
        22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16', 39: 'MXFP4'}


class R:
    def __init__(self, b): self.b, self.o = b, 0
    def take(self, f):
        v = struct.unpack_from(f, self.b, self.o); self.o += struct.calcsize(f); return v[0]
    def s(self):
        n = self.take('<Q'); v = self.b[self.o:self.o + n].decode('utf-8', 'replace'); self.o += n; return v
    def val(self, t):
        if t == 8: return self.s()
        if t == 9:
            et = self.take('<I'); n = self.take('<Q')
            return [self.val(et) for _ in range(n)]
        return self.take(FMT[t])


def dump(path, show_tensors=True):
    r = R(open(path, 'rb').read())
    assert r.take('<I') == 0x46554747
    ver = r.take('<I'); nt = r.take('<Q'); nkv = r.take('<Q')
    print(f'## {path}: v{ver} tensors={nt} kv={nkv}')
    for _ in range(nkv):
        k = r.s(); t = r.take('<I'); v = r.val(t)
        if isinstance(v, list) and len(v) > 64:
            print(f'{k} [{TYPES[t]}] len={len(v)} first={v[:5]!r}')
        elif isinstance(v, str) and len(v) > 300:
            print(f'{k} [str] len={len(v)} start={v[:200]!r}')
        else:
            print(f'{k} = {v!r}')
    if not show_tensors:
        return
    names = []
    for _ in range(nt):
        n = r.s(); nd = r.take('<I'); dims = [r.take('<Q') for _ in range(nd)]; ty = r.take('<I'); off = r.take('<Q')
        names.append((n, dims, GGML.get(ty, ty)))
    # collapse per-layer names
    pat = collections.OrderedDict()
    for n, dims, ty in names:
        parts = n.split('.')
        if parts[0] == 'blk':
            key = 'blk.N.' + '.'.join(parts[2:])
            pat.setdefault(key, []).append((int(parts[1]), dims, ty))
        else:
            print(f'{n} {dims} {ty}')
    for key, items in pat.items():
        shapes = collections.Counter((tuple(d), t) for _, d, t in items)
        layers = sorted(i for i, _, _ in items)
        print(f'{key}: layers {layers[0]}..{layers[-1]} (n={len(layers)}) ' + '; '.join(f'{list(s)} {t} x{c}' for (s, t), c in shapes.items()))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        try:
            dump(p)
        except Exception as e:
            print('ERR', p, e)
