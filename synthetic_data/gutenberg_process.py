# https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv

import polars as pl
from huggingface_hub import snapshot_download
import os


def get_gutenberg_subset(n_shards: int = 1) -> pl.DataFrame:
    """Gutenberg dataset is large, so download only certain shards."""
    gutenberg_location = snapshot_download("SaylorTwift/Gutenberg", repo_type="dataset")
    files = os.listdir(os.path.join(gutenberg_location, "data"))
    files.sort()
    out_pl = None
    for shard_idx in range(n_shards):
        file = files[shard_idx]
        df = pl.read_parquet(os.path.join(gutenberg_location, "data", file))
        if out_pl is None:
            out_pl = df
        else:
            out_pl = out_pl.vstack(df)
    assert out_pl is not None, "could not find any shards"
    return out_pl


def process_gutenberg():
    """Retrieve the catalog, and use it to filter the Gutenberg dataset by."""

    catalog_df = pl.read_csv(
        "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
    )


if __name__ == "__main__":
    process_gutenberg()
