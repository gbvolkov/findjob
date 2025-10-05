import pandas as pd

# df1 and df2 already exist ---------------------------------------------
# Columns:  question | theme | answer
# -----------------------------------------------------------------------

df1 = pd.read_csv("data/answers.csv")
df2 = pd.read_csv("data/answers_changed.csv")

### 1) Same question, different answer  ##################################

diff_ans = (
    df1
    .merge(df2, on=["question", "theme"], how="inner",
           suffixes=("_df1", "_df2"))
    .query("answer_df1 != answer_df2")
    .reset_index(drop=True)
)

diff_ans.to_csv("data/answers_diff.csv", index=False)

# -----------------------------------------------------------------------

### 2) Questions in df2 but NOT in df1  ##################################

only_in_df2 = df2.loc[~df2["question"].isin(df1["question"])]\
                 .reset_index(drop=True)


only_in_df2.to_csv("data/questions_added.csv", index=False)

# -----------------------------------------------------------------------

### 3) Questions in df1 but NOT in df2  ##################################

only_in_df1 = df1.loc[~df1["question"].isin(df2["question"])]\
                 .reset_index(drop=True)
only_in_df1.to_csv("data/questions_removed.csv", index=False)