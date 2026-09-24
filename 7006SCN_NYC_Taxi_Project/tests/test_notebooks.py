"""Validate the reference-style notebooks WITHOUT pretending their results ran."""
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {1: 10, 2: 10, 3: 10, 4: 10}
NUMBERS = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣",
           7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟"}


def test_four_notebooks_parse_and_code_cells_compile():
    notebooks = [ROOT / "notebooks" / f"Task{i}.ipynb" for i in (1, 2, 3, 4)]
    assert all(path.is_file() for path in notebooks)
    for path in notebooks:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        code = [cell.source for cell in book.cells if cell.cell_type == "code"]
        assert len(code) >= 9
        for index, source in enumerate(code):
            compile(source, f"{path.name}:cell-{index}", "exec")
        joined = "\n".join(code)
        assert "make_spark(cfg" in joined
        assert any(phrase in joined for phrase in
                   ("run_task1(", "run_task2(", "run_task3(", "run_task4("))
        assert "84,368,012" not in joined and "0.8689" not in joined
        assert "collision_parquet" not in joined and "RegressionEvaluator" not in joined


def test_numbered_reference_style_is_clear_and_outputs_are_unexecuted():
    for task, count in COUNTS.items():
        book = nbformat.read(ROOT / "notebooks" / f"Task{task}.ipynb", as_version=4)
        assert len(book.cells) == 2 * count + 2
        assert book.cells[0].cell_type == book.cells[-1].cell_type == "markdown"
        for number in range(1, count + 1):
            heading, code = book.cells[2 * number - 1:2 * number + 1]
            assert heading.cell_type == "markdown"
            assert heading.source.startswith(f"## {NUMBERS[number]} ")
            assert code.cell_type == "code" and code.source.strip()
            assert code.execution_count is None and code.outputs == []
    sources = {}
    for task in COUNTS:
        book = nbformat.read(ROOT / "notebooks" / f"Task{task}.ipynb", as_version=4)
        sources[task] = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
    assert "hadoop_file_sizes" in sources[1] and "run_task1" in sources[1]
    assert "VectorAssembler" in sources[1] and "split_processed" in sources[1]
    assert "CrossValidator" in sources[2] and "PipelineModel.load" in sources[2]
    assert "run_task3" in sources[3] and "task3_spark_ui.png" in sources[3]
    assert "run_task4" in sources[4]
    assert 'f"tableau_dashboard_{i}.png"' in sources[4]


def test_python_entry_points_show_plan_without_fabricated_results():
    for script in (ROOT / "NYC_Taxi_TR04.py", ROOT / "scripts" / "run_project.py"):
        shown = subprocess.run([sys.executable, str(script), "--plan"],
                               text=True, capture_output=True, check=True, cwd=ROOT).stdout
        assert "1️⃣  SPARK SESSION CONFIGURATION" in shown
        assert "FOUR CROSSVALIDATORS" in shown
        assert "MODEL SERIALIZATION" in shown
        assert "four-dashboard" in shown
        assert "NOT measured results" in shown
