from pathlib import Path


def save_figure(fig, output_path, dpi=300, bbox_inches="tight"):
    """Save a matplotlib figure, ensuring output directory exists."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
