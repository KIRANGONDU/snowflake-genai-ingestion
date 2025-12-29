import pandas as pd
from transformations import split_fullname, standardize_dob, add_load_date

df = pd.DataFrame({
    "fullname": ["John Doe", "Rahul Kumar", None],
    "dob": ["12-05-1990", "15-Aug-1988", "1992/07/22"]
})

df["firstname"], df["lastname"] = zip(*df["fullname"].apply(split_fullname))
df["dob_std"] = df["dob"].apply(standardize_dob)
df = add_load_date(df)

print(df)
