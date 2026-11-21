from datasets import Dataset
from rich import print
from rich.console import Console
from rich.text import Text

from arc.paths import DATASET_DIR
from evaluate import load_data


COLORS = [
    "black", "red", "green", "yellow", "blue",
    "magenta", "cyan", "white", "bright_red", "bright_green",
]
console = Console()


def render_grid(grid):
    for row in grid:
        visual = Text()
        for cell in row:
            visual.append("  ", style=f"on {COLORS[cell]}")
        visual.append("  " + str(row))
        console.print(visual)


def main():
    df = load_data(DATASET_DIR, sample_count=10)
    dataset = Dataset.from_pandas(df).shuffle(42).select(range(4))
    example = dataset[0]

    print("-----Dataset Statistics-----")
    print(dataset)
    print("-----Train Question Example-----")
    print(example["train"])
    print("-----Test Question Example-----")
    print(example["test"])
    for label, grids in (("Train", example["train"]), ("Test", example["test"])):
        print(f"{label} Input")
        render_grid(grids[0]["input"])
        print(f"{label} Output")
        render_grid(grids[0]["output"])


if __name__ == "__main__":
    main()
