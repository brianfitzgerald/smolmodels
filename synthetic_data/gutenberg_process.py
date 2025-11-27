import polars as pl
from huggingface_hub import snapshot_download
import os
from dataclasses import dataclass

import fire
import trio
import httpx
from loguru import logger


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


EXCLUDED_CATEGORIES = {
    "Romance",
    "Crime, Thrillers and Mystery",
    "Essays, Letters & Speeches",
    "Poetry",
    "British Literature",
    "Biographies",
    "Mythology, Legends & Folklore",
    "Travel Writing",
    "Plays/Films/Dramas",
    "Classics of Literature",
    "History - American",
    "Journals",
    "Sports/Hobbies",
    "Philosophy & Ethics",
    "Religion/Spirituality",
    "History - Warfare",
    "Politics",
    "History - Modern (1750+)",
    "Journalism/Media/Writing",
    "Gender & Sexuality Studies",
}


def parse_bookshelves(bookshelves_str: str | None) -> dict[str, list[str]]:
    """Parse a bookshelves string into a dict of category -> list of values."""
    bookshelves_dict: dict[str, list[str]] = {}
    if not bookshelves_str:
        return bookshelves_dict

    # Split by semicolon to get individual entries
    entries = bookshelves_str.split(";")
    for entry in entries:
        entry = entry.strip()
        if ":" in entry:
            # Split by colon to separate category from value
            category, value = entry.split(":", 1)
            category = category.strip()
            value = value.strip()

            # Add to dict, creating list if category doesn't exist
            if category not in bookshelves_dict:
                bookshelves_dict[category] = []
            bookshelves_dict[category].append(value)
        else:
            # Handle entries without a category (like "Nobel Prizes in Literature")
            if entry:  # Skip empty entries
                if "Other" not in bookshelves_dict:
                    bookshelves_dict["Other"] = []
                bookshelves_dict["Other"].append(entry)

    return bookshelves_dict


# https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
def get_catalog_ids() -> list[int]:
    """Retrieve the catalog, and use it to filter the Gutenberg dataset by genre and language."""

    catalog_df = pl.read_csv(
        "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
    )
    catalog_df = catalog_df.filter(pl.col("Bookshelves").is_not_null())
    catalog_df = catalog_df.filter(pl.col("Language") == "en")
    catalog_df = catalog_df.with_columns(
        pl.col("Bookshelves")
        .map_elements(parse_bookshelves, return_dtype=pl.Object)
        .alias("bookshelves_parsed")
    )
    catalog_df = catalog_df.filter(
        pl.col("bookshelves_parsed").map_elements(
            lambda x: (
                "Category" in x
                and (
                    any(
                        "Science-Fiction" in val or "Fantasy" in val
                        for val in x["Category"]
                    )
                    or any("American Literature" in val for val in x["Category"])
                )
                and not any(val in EXCLUDED_CATEGORIES for val in x["Category"])
            ),
            return_dtype=pl.Boolean,
        )
    )

    catalog_df = catalog_df.rename({"Text#": "id"})

    return catalog_df["id"].to_list()


@dataclass
class GutenbergBook:
    id: int
    text: str | None
    error: str | None = None


# Gutenberg rate limit: be respectful, limit concurrent requests
MAX_CONCURRENT_REQUESTS = 10
REQUEST_DELAY_SECONDS = 0.1  # Small delay between starting requests


def get_book_url(book_id: int) -> str:
    """Get the URL for a Gutenberg book's plain text."""
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


async def fetch_single_book(
    client: httpx.AsyncClient,
    book_id: int,
    semaphore: trio.Semaphore,
) -> GutenbergBook:
    """Fetch a single book from Project Gutenberg, respecting rate limits."""
    async with semaphore:
        url = get_book_url(book_id)
        try:
            response = await client.get(url, follow_redirects=True)
            if response.status_code == 200:
                return GutenbergBook(id=book_id, text=response.text)
            else:
                return GutenbergBook(
                    id=book_id,
                    text=None,
                    error=f"HTTP {response.status_code}",
                )
        except Exception as e:
            return GutenbergBook(id=book_id, text=None, error=str(e))
        finally:
            # Small delay after each request to be respectful of rate limits
            await trio.sleep(REQUEST_DELAY_SECONDS)


async def fetch_all_books(book_ids: list[int]) -> list[GutenbergBook]:
    """Fetch all books concurrently using trio, respecting rate limits."""
    results: list[GutenbergBook] = []
    semaphore = trio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def fetch_and_store(book_id: int) -> None:
            result = await fetch_single_book(client, book_id, semaphore)
            results.append(result)
            if result.error:
                logger.warning(f"Failed to fetch book {book_id}: {result.error}")
            else:
                logger.info(f"Fetched book {book_id} ({len(result.text or '')} chars)")

        async with trio.open_nursery() as nursery:
            for book_id in book_ids:
                nursery.start_soon(fetch_and_store, book_id)

    return results


def process_gutenberg(
    output_path: str = "dataset_files/gutenberg_books.parquet",
) -> None:
    """Process the Gutenberg dataset by filtering by genre and language, then downloading books."""
    logger.info("Fetching catalog IDs...")
    catalog_ids = get_catalog_ids()
    logger.info(f"Found {len(catalog_ids)} books matching criteria")

    logger.info("Fetching book contents...")
    books = trio.run(fetch_all_books, catalog_ids)

    # Filter successful downloads and create DataFrame
    successful_books = [b for b in books if b.text is not None]
    logger.info(
        f"Successfully downloaded {len(successful_books)}/{len(catalog_ids)} books"
    )

    df = pl.DataFrame(
        {
            "id": [b.id for b in successful_books],
            "text": [b.text for b in successful_books],
        }
    )

    df.write_parquet(output_path)
    logger.info(f"Saved {len(df)} books to {output_path}")


if __name__ == "__main__":
    fire.Fire(process_gutenberg)
