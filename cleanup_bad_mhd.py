import SimpleITK as sitk
from pathlib import Path

d = Path(r"E:\lung_ct_lesion_dataset\LUNA16-New\retina_mhd")
bad = []

for p in d.glob("*.mhd"):
    try:
        sitk.ReadImage(str(p))
    except Exception:
        bad.append(p)

for p in bad:
    for q in (p, p.with_suffix(".raw"), p.with_suffix(".zraw")):
        if q.exists():
            q.unlink()

print("bad_removed =", len(bad))
