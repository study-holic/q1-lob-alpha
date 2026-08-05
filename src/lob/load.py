def load_lobster(message_path, orderbook_path, levels=10) -> pd.DataFrame:
    """Load and join the two files into one DataFrame.
    Prices converted to dollars, time converted to something readable.
    Raises if the files don't align or the schema is wrong."""

def validate_book(df) -> None:
    """Raise on any row that violates a book invariant."""
