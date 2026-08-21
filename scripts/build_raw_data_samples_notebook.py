from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src" / "raw_data_samples.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nb["cells"] = [
    md(
        """
        # Raw MovieLens 100K And OGBL-Collab Samples

        This notebook shows what the raw input files look like before the preprocessing in the experiment suite.

        It focuses on:

        - raw `MovieLens 100K` ratings rows
        - raw `ogbl-collab` edge, edge-weight, edge-year, and node-feature files
        - small samples, shapes, and column names so the source data is easier to understand
        """
    ),
    code(
        """
        from pathlib import Path

        import pandas as pd

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 160)

        # Resolve the src directory robustly whether the kernel starts in the
        # repo root or inside the src folder.
        cwd = Path.cwd().resolve()
        if (cwd / "data").exists():
            SRC_DIR = cwd
        elif (cwd / "src" / "data").exists():
            SRC_DIR = cwd / "src"
        elif (cwd.parent / "src" / "data").exists():
            SRC_DIR = cwd.parent / "src"
        else:
            raise FileNotFoundError("Could not locate the src/data directory from the current working directory.")

        DATA_DIR = SRC_DIR / "data"
        MOVIELENS_DIR = DATA_DIR / "movielens_100k" / "ml-100k"
        OGB_RAW_DIR = DATA_DIR / "ogb" / "ogbl_collab" / "raw"

        print(f"SRC_DIR: {SRC_DIR}")
        print(f"MovieLens dir exists: {MOVIELENS_DIR.exists()}")
        print(f"OGB raw dir exists: {OGB_RAW_DIR.exists()}")
        """
    ),
    md(
        """
        ## MovieLens 100K Raw Ratings

        The raw `u.data` file is tab-separated and stores:

        - `user_id`
        - `item_id`
        - `rating`
        - `timestamp`
        """
    ),
    code(
        """
        movielens_raw = pd.read_csv(
            MOVIELENS_DIR / "u.data",
            sep="\\t",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        print("MovieLens raw shape:", movielens_raw.shape)
        movielens_raw.head(10)
        """
    ),
    code(
        """
        # Show a random sample as well so the data does not look overly sequential.
        movielens_raw.sample(10, random_state=42)
        """
    ),
    code(
        """
        movielens_raw.describe(include="all")
        """
    ),
    md(
        """
        ## OGBL-Collab Raw Files

        The `ogbl-collab` raw directory is split across several files:

        - `edge.csv.gz`: source-target node pairs
        - `edge_weight.csv.gz`: edge weights aligned row-for-row with `edge.csv.gz`
        - `edge_year.csv.gz`: years aligned row-for-row with `edge.csv.gz`
        - `node-feat.csv.gz`: node feature matrix

        The first three files should be read together because row `i` in each file refers to the same collaboration edge.
        """
    ),
    code(
        """
        ogb_edges = pd.read_csv(OGB_RAW_DIR / "edge.csv.gz", header=None, names=["source_node", "target_node"])
        ogb_edge_weight = pd.read_csv(OGB_RAW_DIR / "edge_weight.csv.gz", header=None, names=["edge_weight"])
        ogb_edge_year = pd.read_csv(OGB_RAW_DIR / "edge_year.csv.gz", header=None, names=["edge_year"])

        print("OGB edge shape:", ogb_edges.shape)
        print("OGB edge_weight shape:", ogb_edge_weight.shape)
        print("OGB edge_year shape:", ogb_edge_year.shape)

        ogb_edge_sample = pd.concat(
            [ogb_edges.head(10), ogb_edge_weight.head(10), ogb_edge_year.head(10)],
            axis=1,
        )
        ogb_edge_sample
        """
    ),
    code(
        """
        # This merged sample makes the row alignment explicit.
        sample_idx = ogb_edges.sample(10, random_state=42).index
        ogb_edge_sample_random = pd.concat(
            [
                ogb_edges.loc[sample_idx].reset_index(names="row_id"),
                ogb_edge_weight.loc[sample_idx].reset_index(drop=True),
                ogb_edge_year.loc[sample_idx].reset_index(drop=True),
            ],
            axis=1,
        )
        ogb_edge_sample_random
        """
    ),
    code(
        """
        ogb_node_feat = pd.read_csv(OGB_RAW_DIR / "node-feat.csv.gz", header=None)

        print("OGB node feature matrix shape:", ogb_node_feat.shape)
        print("First 5 rows and first 10 feature columns:")
        ogb_node_feat.iloc[:5, :10]
        """
    ),
    code(
        """
        # A compact summary of the raw files available for ogbl-collab.
        raw_file_summary = pd.DataFrame(
            [
                {"file": "edge.csv.gz", "rows": len(ogb_edges), "columns": ogb_edges.shape[1]},
                {"file": "edge_weight.csv.gz", "rows": len(ogb_edge_weight), "columns": ogb_edge_weight.shape[1]},
                {"file": "edge_year.csv.gz", "rows": len(ogb_edge_year), "columns": ogb_edge_year.shape[1]},
                {"file": "node-feat.csv.gz", "rows": len(ogb_node_feat), "columns": ogb_node_feat.shape[1]},
            ]
        )
        raw_file_summary
        """
    ),
    md(
        """
        ## Notes

        - `MovieLens 100K` starts as a straightforward ratings table.
        - `ogbl-collab` starts as multiple aligned graph files rather than one single table.
        - The experiment notebook transforms both datasets into a common edge-prediction format so the same recommenders and evaluation code can be reused.
        """
    ),
]

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK.open("w", encoding="utf-8") as handle:
    nbf.write(nb, handle)

print(f"Wrote notebook to {NOTEBOOK}")
