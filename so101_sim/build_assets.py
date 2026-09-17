"""Generate procedural assets: wood table texture (PNG) and hollow cup visual meshes (STL).

Run once: `python -m so101_sim.build_assets`. Outputs are committed under so101_sim/assets/generated/.
No third-party dependencies beyond numpy.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

GENERATED = Path(__file__).parent / "assets" / "generated"


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def wood_texture(size: int = 1024, seed: int = 7) -> np.ndarray:
    """Light oak plank texture similar to the reference photo (planks run along the image x axis)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / size
    # Low-frequency wobble so grain lines are not perfectly straight.
    wobble = 0.0
    for k, amp in ((3, 0.02), (7, 0.008), (13, 0.003)):
        wobble = wobble + amp * np.sin(2 * np.pi * (k * x + rng.uniform(0, 1))) * np.cos(2 * np.pi * (k * 0.7 * y + rng.uniform(0, 1)))
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * (x * 14 + wobble * 4))
    grain = grain ** 4  # thin dark lines, wide light gaps
    fine = rng.normal(0, 1, (size, size))
    fine = np.clip((fine + np.roll(fine, 1, 1) + np.roll(fine, 2, 1)) / 3, -2, 2) * 0.03
    # Plank seams every 1/4 of the texture height.
    seam = (np.abs(((x * 3) % 1.0) - 0.5) > 0.497).astype(np.float64)
    base = np.array([196, 160, 112]) / 255.0
    dark = np.array([140, 100, 62]) / 255.0
    color = base[None, None, :] * (1 - 0.45 * grain[..., None]) + dark[None, None, :] * (0.45 * grain[..., None])
    color = color * (1 + fine[..., None]) * (1 - 0.35 * seam[..., None])
    return (np.clip(color, 0, 1) * 255).astype(np.uint8)


def write_stl(path: Path, triangles: np.ndarray) -> None:
    """triangles: (N, 3, 3) float array, vertices counter-clockwise seen from the visible side."""
    n = len(triangles)
    with path.open("wb") as f:
        f.write(b"\x00" * 80 + struct.pack("<I", n))
        for tri in triangles:
            a, b, c = tri
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.zeros(3)
            f.write(struct.pack("<3f", *normal))
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(b"\x00\x00")


def frustum(r_bottom: float, r_top: float, z0: float, z1: float, segments: int, outward: bool) -> list:
    tris = []
    for i in range(segments):
        a0 = 2 * np.pi * i / segments
        a1 = 2 * np.pi * (i + 1) / segments
        b0 = np.array([r_bottom * np.cos(a0), r_bottom * np.sin(a0), z0])
        b1 = np.array([r_bottom * np.cos(a1), r_bottom * np.sin(a1), z0])
        t0 = np.array([r_top * np.cos(a0), r_top * np.sin(a0), z1])
        t1 = np.array([r_top * np.cos(a1), r_top * np.sin(a1), z1])
        if outward:
            tris += [(b0, b1, t1), (b0, t1, t0)]
        else:
            tris += [(b0, t1, b1), (b0, t0, t1)]
    return tris


def disc(radius: float, z: float, segments: int, facing_up: bool, r_inner: float = 0.0) -> list:
    tris = []
    for i in range(segments):
        a0 = 2 * np.pi * i / segments
        a1 = 2 * np.pi * (i + 1) / segments
        o0 = np.array([radius * np.cos(a0), radius * np.sin(a0), z])
        o1 = np.array([radius * np.cos(a1), radius * np.sin(a1), z])
        i0 = np.array([r_inner * np.cos(a0), r_inner * np.sin(a0), z])
        i1 = np.array([r_inner * np.cos(a1), r_inner * np.sin(a1), z])
        if facing_up:
            tris += [(i0, o0, o1), (i0, o1, i1)]
        else:
            tris += [(i0, o1, o0), (i0, i1, i1 * 0 + i1)]  # second tri degenerate when r_inner == 0
    return [t for t in tris if np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-12]


def cup_meshes(height: float, r_top: float, r_bottom: float, wall: float, floor: float, segments: int = 48):
    """Return (outer_triangles, inner_triangles) for an open-topped tapered cup, base centered at origin."""
    outer = frustum(r_bottom, r_top, 0.0, height, segments, outward=True)
    outer += disc(r_bottom, 0.0, segments, facing_up=False)
    outer += disc(r_top, height, segments, facing_up=True, r_inner=r_top - wall)  # rim
    inner_r_bottom = r_bottom - wall + (r_top - r_bottom) * (floor / height)
    inner = frustum(inner_r_bottom, r_top - wall, floor, height, segments, outward=False)
    inner += disc(inner_r_bottom, floor, segments, facing_up=True)
    return np.array(outer), np.array(inner)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    write_png(GENERATED / "wood.png", wood_texture())
    # Cup meshes are generated on demand for the configured dimensions by scene.cup_mesh_files().
    from .scene import SceneConfig, cup_mesh_files
    cup_mesh_files(SceneConfig())
    print("wrote", sorted(p.name for p in GENERATED.iterdir()))


if __name__ == "__main__":
    main()
