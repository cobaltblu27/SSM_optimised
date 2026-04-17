import pandas as pd
import os
import argparse
from pathlib import Path

def parse_args():
    # Default: project_root/results (works regardless of where script is run from)
    project_root = Path(__file__).parent.parent
    default_result_root = project_root / "results"
    default_output_dir = project_root / "logs"

    parser = argparse.ArgumentParser(description="Collect best AUC iteration for each target")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value used in file paths")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the resulting CSV (default: logs/target_max_AUC_dict_<threshold>.csv)")
    parser.add_argument("--result_root", type=str, default=str(default_result_root), help="Root directory for results (default: <project_root>/results)")
    parser.add_argument("--targets", nargs="+", default=[
        "ACES", "ADA", "ANDR", "EGFR", "FA10",
        "KIT", "PLK1", "SRC", "THRB", "UROK"
    ], help="List of targets to process")
    args = parser.parse_args()

    # Auto-generate output_path if not provided
    if args.output_path is None:
        args.output_path = str(default_output_dir / f"target_max_AUC_dict_{args.threshold}.csv")

    return args

def main():
    args = parse_args()
    threshold = args.threshold
    output_path = args.output_path
    result_root = args.result_root
    targets = args.targets

    target_max_AUC_dict = {}

    for target in targets:
        try:
            path = f"{result_root}/{target}/performance.csv"
            df = pd.read_csv(path)
            max_iter = df.loc[df["AUC"].idxmax(), "iteration"]
            target_max_AUC_dict[target] = int(max_iter)
        except Exception as e:
            print(f"Failed to load performance for {target}: {e}")
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(target_max_AUC_dict.items(), columns=["target", "max_AUC_iteration"]).to_csv(output_path, index=False)
    print(f"Best AUC iteration saved to {output_path}")

if __name__ == "__main__":
    main()
