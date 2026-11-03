import pandas as pd

df = pd.read_csv("data/evaluation/final_evaluation.csv")

print("Total questions:", len(df))

print("\nCorrectness distribution:")
print(df["Correctness"].value_counts().sort_index())

print("\nFaithfulness distribution:")
print(df["Faithfulness"].value_counts().sort_index())

print("\nAverage scores:")
print(df[["Correctness", "Faithfulness"]].mean().round(2))

print("\nQuestion-wise scores:")
print(
    df[
        ["Question ID", "Correctness", "Faithfulness"]
    ].to_string(index=False)
)