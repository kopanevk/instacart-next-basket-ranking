import json
from pathlib import Path


def test_retrieval_notebook_keeps_final_labels_closed():
    path = Path("notebooks/02_baselines_and_candidates.ipynb")
    notebook = json.loads(path.read_text())
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "order_products__train" not in code
    assert "history_items(orders, prior_items, examples, 'ranker')" in code
    assert "assert_no_target_in_history(history, examples, 'ranker')" in code

