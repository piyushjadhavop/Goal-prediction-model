import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("Data.csv")
    max_goals = np.max(df["Goals"].to_numpy())
    print(f"Max goals: {max_goals}")


if __name__ == "__main__":
    main()
