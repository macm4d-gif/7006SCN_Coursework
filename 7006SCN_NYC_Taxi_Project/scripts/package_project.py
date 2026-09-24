"""Build the explicitly SOURCE-ONLY downloadable ZIP; reject private/data artefacts.

Never ship actual student config, TLC raw/processed data, Spark models or test
fixtures as assessed results. The official answer sheet is Word OOXML even
though the user-uploaded original had a .pdf suffix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / f"{ROOT.name}.zip"
SINGLE = {
    ".gitignore", "NYC_Taxi_TR04.py", "PROJECT_AUDIT.md", "README.md",
    "REFERENCE_ALIGNMENT.md", "SUBMISSION_STATUS.md", "TESTING.md", "requirements.txt",
    "report/REPORT_WORKSHEET.md", "report/author_content.example.json",
    "report/7006SCN_Official_Answer_Sheet_TEMPLATE.docx",
    "report/TR04_Official_Answer_Sheet_WORKING_INCOMPLETE.docx",
    "tableau/BUILD_DASHBOARDS.md", "artifacts/.gitkeep", "results/.gitkeep",
    "data/raw/.gitkeep", "config/config.example.json", "config/config.university.example.json",
}


def selected_files() -> list[Path]:
    files = set(SINGLE)
    for folder in ("coursework", "scripts", "tests"):
        files.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).glob("*.py"))
    files.update(f"notebooks/Task{i}.ipynb" for i in range(1, 5))
    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required controlled source files are missing: {sorted(missing)}")
    for i in range(1, 5):
        book = json.loads((ROOT / "notebooks" / f"Task{i}.ipynb").read_text(encoding="utf-8"))
        if any(c.get("outputs") or c.get("execution_count") is not None
               for c in book["cells"] if c["cell_type"] == "code"):
            raise ValueError("A notebook now has output. Inspect it manually before packaging real/personal results.")
    return [ROOT / name for name in sorted(files)]


def build(destination: Path = DESTINATION) -> dict:
    files = selected_files()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            arcname = f"{ROOT.name}/{path.relative_to(ROOT)}"
            info = ZipInfo(str(arcname), date_time=(2026, 9, 24, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            z.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    with ZipFile(destination) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names)) == len(files)
        assert all("/config/config.json" not in n and "/data/processed/" not in n
                   and "/results/task" not in n and "/.venv/" not in n
                   for n in names)
    return {"path": str(destination), "controlled_files": len(files),
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(build())
