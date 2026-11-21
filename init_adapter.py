import os

from arc import ARCSolver


def main():
    solver = ARCSolver(token=os.environ.get("HF_TOKEN"))
    solver.initialize_fresh_adapter()


if __name__ == "__main__":
    main()
