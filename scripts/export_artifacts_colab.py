# Run this in a Colab cell, signed in to the Google account that holds MyDrive/bbnet_repro.
# It packages everything that belongs in the repository's data/ folder into MyDrive/bbnet_repo_data.zip.
from google.colab import drive
drive.mount("/content/drive")
import os, glob, zipfile

SRC = "/content/drive/MyDrive/bbnet_repro"
OUT = "/content/drive/MyDrive/bbnet_repo_data.zip"
files = [os.path.join(SRC, n) for n in ("dataset.json", "bench.jsonl", "cert.jsonl", "table7.jsonl",
                                        "relabel_groups.json", "relabel.jsonl") if os.path.exists(os.path.join(SRC, n))]
files += glob.glob(f"{SRC}/runs/**/history.json", recursive=True)
files += glob.glob(f"{SRC}/runs/**/test_eval.json", recursive=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        z.write(p, os.path.join("data", os.path.relpath(p, SRC)))
        print(f"{os.path.relpath(p, SRC):60s} {os.path.getsize(p) / 1e6:7.1f} MB")
print(f"\nCreated {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB, {len(files)} files)")
print("Attach the main checkpoint to a GitHub Release instead of committing it:")
print(f"  {SRC}/runs/main_tanner_aug1_canon1_film1_s0/best.pt")
