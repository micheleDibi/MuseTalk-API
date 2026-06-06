"""Pre-fetch the two weights that the upstream MuseTalk pipeline downloads
LAZILY at first runtime use:

1. GFPGANv1.4.pth — fetched by ``api.inference_service.MuseTalkInference._load_gfpgan``
   on the first call with ``--enhance``. Hard-coded GitHub release URL.
2. s3fd-619a316812.pth — fetched by the ``face_alignment`` library inside
   ``musetalk.utils.preprocessing.FaceAlignment(...)`` on the first face
   detection call. Hosted at adrianbulat.com.

On compute nodes without outbound internet (MN5, Leonardo, etc.) those lazy
downloads fail with a misleading socket error halfway through a 15-min run.

Run this script INSIDE the Singularity build's ``%post`` stage (where the
login node still has internet). The files land at the paths the runtime
code already looks for:

    /app/models/gfpgan/GFPGANv1.4.pth
    /app/models/torch_hub/checkpoints/s3fd-619a316812.pth   (via TORCH_HOME)

At runtime the engine sees ``GFPGAN_MODEL_PATH`` (set by ``%environment``)
and uses the local file; the face_alignment library finds s3fd under
``$TORCH_HOME/checkpoints/`` and skips the download.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


WEIGHTS = [
    {
        "name": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "dest": Path("/app/models/gfpgan/GFPGANv1.4.pth"),
        "min_size_mb": 300,  # actual ~333 MB; sanity floor
    },
    {
        "name": "s3fd-619a316812.pth",
        "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "dest": Path("/app/models/torch_hub/checkpoints/s3fd-619a316812.pth"),
        "min_size_mb": 80,  # actual ~86 MB
    },
]


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prefetch] downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    size_bytes = dest.stat().st_size
    print(f"[prefetch]   wrote {size_bytes / 1e6:.1f} MB", flush=True)
    return size_bytes


def main() -> int:
    rc = 0
    for entry in WEIGHTS:
        name = entry["name"]
        try:
            size_bytes = _download(entry["url"], entry["dest"])
        except Exception as e:
            print(f"[prefetch] FAILED {name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            rc = 1
            continue
        size_mb = size_bytes / 1e6
        if size_mb < entry["min_size_mb"]:
            print(
                f"[prefetch] WARNING: {name} is only {size_mb:.1f} MB "
                f"(expected >= {entry['min_size_mb']} MB) — file may be "
                f"truncated/corrupted",
                file=sys.stderr,
            )
            rc = 2

    if rc == 0:
        print("[prefetch] all lazy weights baked into the image", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
