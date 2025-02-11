import pandas as pd

pathname = "/Users/zach/Downloads/pert_flags.csv"


cols = []
df = pd.read_csv(pathname, header=None)
for index, row in df.iterrows():
    for i, value in enumerate(row):
        if value == 1:
            cols.append(i)
            print(f"Row {index}, Column {i}")


print(len(cols))
print(len(set(cols)))
