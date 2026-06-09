"""Tiny QR-code encoder — just enough to turn a short LAN URL into a scannable
code, with no new dependency (numpy only, in the project's no-deps spirit).

Scope: byte mode, error-correction level L, versions 1-4 (21x21 .. 33x33), a
single error-correction block. That holds up to 80 bytes — any
``http://<ip>:<port>/listen`` URL fits with room to spare. Output is a boolean
module matrix (True = dark) or an SVG string.

The full QR standard (multi-block interleaving, version-info modules for v>=7,
the other EC levels) is intentionally left out — we never need it here. The
encoder is verified against OpenCV's QRCodeDetector in tools/?: feed it a URL,
re-decode the rendered matrix, assert it round-trips.

Reference: ISO/IEC 18004. Generator polynomials and format-info strings are the
standard tabulated values.
"""

from __future__ import annotations

import numpy as np

# ---- GF(256) arithmetic (primitive polynomial 0x11D) ---------------------
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _rs_generator(degree: int) -> list[int]:
    """Generator polynomial for `degree` EC codewords: prod (x - a^i)."""
    g = [1]
    for i in range(degree):
        ng = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            ng[j] ^= c
            ng[j + 1] ^= _gf_mul(c, _EXP[i])
        g = ng
    return g


def _rs_ec(data: list[int], ec_count: int) -> list[int]:
    """Reed-Solomon EC codewords for `data` (polynomial division remainder)."""
    gen = _rs_generator(ec_count)
    rem = [0] * ec_count
    for d in data:
        factor = d ^ rem[0]
        rem = rem[1:] + [0]
        for i in range(ec_count):
            rem[i] ^= _gf_mul(gen[i + 1], factor)
    return rem


# ---- per-version tables (EC level L, single block) -----------------------
# (data codewords, EC codewords per block) for versions 1..4 at level L.
_VERSION_L = {
    1: (19, 7),
    2: (34, 10),
    3: (55, 15),
    4: (80, 20),
}
# Alignment-pattern centre for v2..4 (single extra pattern at (c, c); v1 none).
_ALIGN_CENTRE = {2: 18, 3: 22, 4: 26}
# Format-info bit strings (15 bits) for EC level L, mask 0..7 — standard table,
# already BCH-encoded and XORed with the 0x5412 mask.
_FORMAT_L = [
    0b111011111000100, 0b111001011110011, 0b111110110101010, 0b111100010011101,
    0b110011000101111, 0b110001100011000, 0b110110001000001, 0b110100101110110,
]


def _smallest_version(n_bytes: int) -> int:
    for v in (1, 2, 3, 4):
        if n_bytes <= _VERSION_L[v][0] - 2:   # minus mode(1) + length(1) overhead
            return v
    raise ValueError(f"data too long for QR v1-4 ({n_bytes} bytes)")


def _bitstream(data: bytes, version: int) -> list[int]:
    cap_bytes = _VERSION_L[version][0]
    bits: list[int] = []

    def put(val: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            bits.append((val >> i) & 1)

    put(0b0100, 4)              # byte mode
    put(len(data), 8)           # char count (8-bit for byte mode, v1-9)
    for b in data:
        put(b, 8)
    # terminator (up to 4 zero bits) + pad to byte boundary
    put(0, min(4, cap_bytes * 8 - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    # pad bytes 0xEC, 0x11 alternating up to capacity
    pad = [0xEC, 0x11]
    i = 0
    while len(bits) < cap_bytes * 8:
        put(pad[i % 2], 8)
        i += 1
    # to codewords
    return [int("".join(str(b) for b in bits[k:k + 8]), 2)
            for k in range(0, len(bits), 8)]


# ---- module matrix --------------------------------------------------------
def _finder(grid, fn, r0, c0) -> None:
    for r in range(-1, 8):
        for c in range(-1, 8):
            rr, cc = r0 + r, c0 + c
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                continue
            fn[rr, cc] = True
            inb = 0 <= r <= 6 and 0 <= c <= 6
            ring = r in (0, 6) or c in (0, 6)
            core = 2 <= r <= 4 and 2 <= c <= 4
            grid[rr, cc] = inb and (ring or core)


def _build_matrix(codewords: list[int], version: int, mask: int) -> np.ndarray:
    size = 4 * version + 17
    grid = np.zeros((size, size), dtype=bool)
    fn = np.zeros((size, size), dtype=bool)   # function-module mask

    # finders + separators (separators handled by the -1..7 border above)
    _finder(grid, fn, 0, 0)
    _finder(grid, fn, 0, size - 7)
    _finder(grid, fn, size - 7, 0)

    # timing patterns
    for i in range(8, size - 8):
        v = (i % 2 == 0)
        grid[6, i] = v; fn[6, i] = True
        grid[i, 6] = v; fn[i, 6] = True

    # alignment pattern (single, for v2..4)
    if version in _ALIGN_CENTRE:
        c = _ALIGN_CENTRE[version]
        for r in range(-2, 3):
            for cc in range(-2, 3):
                ring = max(abs(r), abs(cc))
                grid[c + r, c + cc] = ring != 1
                fn[c + r, c + cc] = True

    # dark module + reserve format-info areas
    grid[4 * version + 9, 8] = True
    fn[4 * version + 9, 8] = True
    for i in range(9):
        if i != 6:
            fn[8, i] = True; fn[i, 8] = True
    for i in range(8):
        fn[8, size - 1 - i] = True
        fn[size - 1 - i, 8] = True

    # data placement: 2-wide columns, right→left, zigzag up/down, skip col 6
    bits = [(cw >> (7 - i)) & 1 for cw in codewords for i in range(8)]
    idx = 0
    up = True
    col = size - 1
    while col > 0:
        if col == 6:
            col -= 1
        for n in range(size):
            row = (size - 1 - n) if up else n
            for c in (col, col - 1):
                if fn[row, c]:
                    continue
                b = bits[idx] if idx < len(bits) else 0
                idx += 1
                if _mask_bit(mask, row, c):
                    b ^= 1
                grid[row, c] = bool(b)
        up = not up
        col -= 2

    # format info (level L + mask), placed in the two reserved strips
    fmt = _FORMAT_L[mask]
    fbits = [(fmt >> (14 - i)) & 1 for i in range(15)]
    # around top-left
    coords1 = [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
               (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8)]
    # top-right + bottom-left copy
    coords2 = [(size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
               (size - 5, 8), (size - 6, 8), (size - 7, 8),
               (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5),
               (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1)]
    for (r, c), b in zip(coords1, fbits):
        grid[r, c] = bool(b)
    for (r, c), b in zip(coords2, fbits):
        grid[r, c] = bool(b)
    return grid


def _mask_bit(mask: int, r: int, c: int) -> bool:
    if mask == 0: return (r + c) % 2 == 0
    if mask == 1: return r % 2 == 0
    if mask == 2: return c % 3 == 0
    if mask == 3: return (r + c) % 3 == 0
    if mask == 4: return (r // 2 + c // 3) % 2 == 0
    if mask == 5: return (r * c) % 2 + (r * c) % 3 == 0
    if mask == 6: return ((r * c) % 2 + (r * c) % 3) % 2 == 0
    return ((r + c) % 2 + (r * c) % 3) % 2 == 0


# ---- mask penalty (ISO 18004 §8.8.2) --------------------------------------
def _penalty(grid: np.ndarray) -> int:
    size = grid.shape[0]
    score = 0
    # rule 1: runs of >=5 same-colour modules in a row/column
    for line in list(grid) + list(grid.T):
        run = 1
        for i in range(1, size):
            if line[i] == line[i - 1]:
                run += 1
            else:
                if run >= 5:
                    score += 3 + (run - 5)
                run = 1
        if run >= 5:
            score += 3 + (run - 5)
    # rule 2: 2x2 blocks of one colour
    blk = grid[:-1, :-1]
    same = (blk == grid[1:, :-1]) & (blk == grid[:-1, 1:]) & (blk == grid[1:, 1:])
    score += 3 * int(np.count_nonzero(same))
    # rule 3: finder-like 1:1:3:1:1 patterns (dark-light runs) in rows/cols
    pat1 = [True, False, True, True, True, False, True, False, False, False, False]
    pat2 = pat1[::-1]
    for line in list(grid) + list(grid.T):
        ln = list(line)
        for i in range(size - 11):
            seg = ln[i:i + 11]
            if seg == pat1 or seg == pat2:
                score += 40
    # rule 4: proportion of dark modules
    dark = int(np.count_nonzero(grid))
    pct = dark * 100 // (size * size)
    score += 10 * (min(abs(pct - 50) // 5, 20))
    return score


def matrix(data: str) -> np.ndarray:
    """Encode `data` (UTF-8) to a QR module matrix (bool, True = dark), picking
    the smallest version (1-4, level L) that fits and the lowest-penalty mask."""
    raw = data.encode("utf-8")
    version = _smallest_version(len(raw))
    dwords = _bitstream(raw, version)
    ec = _rs_ec(dwords, _VERSION_L[version][1])
    codewords = dwords + ec
    best, best_score = None, None
    for m in range(8):
        g = _build_matrix(codewords, version, m)
        s = _penalty(g)
        if best_score is None or s < best_score:
            best, best_score = g, s
    return best


def svg(data: str, quiet: int = 4, module: int = 8) -> str:
    """Render `data` as a self-contained SVG QR string (dark modules as rects on
    a white field, `quiet`-module border, crisp edges)."""
    g = matrix(data)
    size = g.shape[0]
    dim = size + 2 * quiet
    px = dim * module
    rects = []
    ys, xs = np.where(g)
    for y, x in zip(ys.tolist(), xs.tolist()):
        rects.append(f'<rect x="{(x + quiet) * module}" y="{(y + quiet) * module}" '
                     f'width="{module}" height="{module}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" '
            f'viewBox="0 0 {px} {px}" shape-rendering="crispEdges">'
            f'<rect width="{px}" height="{px}" fill="#fff"/>'
            f'<g fill="#000">{"".join(rects)}</g></svg>')
