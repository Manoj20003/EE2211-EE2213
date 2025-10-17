# %% [markdown]
# ### （i）

# %%
# need to install openpyxl to read .xlsx files"
from pandas import read_excel
dataset = read_excel('synthetic-data.xlsx')
print(dataset)
print(dataset.describe())
# pd.describe(): generate descriptive statistics of the numeric columns in a DataFrame.
print(dataset.info())
# pd.info(): concise summary of a DataFrame, including the number of non-null entries and data types.

# %% [markdown]
# ### （ii）

# %%
# check for missing values
missing_locs = dataset.isnull().to_numpy().nonzero()  # returns a tuple of two arrays: (row_indices, col_indices)
# dataset.isnull(): Returns a Boolean DataFrame: True for missing value, False otherwise
# .to_numpy(): Convert to numpy array of Booleans
#.nonzero(): A Numpy method returns indices of the elements that are non-zero/True.

row_indices = missing_locs[0]
col_indices = missing_locs[1]
for row_index, col_index in zip(row_indices, col_indices): 
                           # zip(): takes two or more iterables (like lists, tuples, arrays) 
                           #        and pairs up their elements by position.
    print(f"Missing value at Row {row_index}, Column '{dataset.columns[col_index]}'")
    # dataset.columns: holds all the column labels of your DataFrame

# %%
#test cell
dataset.isnull()

# %%
#test cell
dataset.isnull().to_numpy()

# %%
#test cell
dataset.isnull().to_numpy().nonzero()

# %%
# check for duplicate entries
duplicates = dataset.duplicated() # Returns a boolean Series marking whether each row is a duplicate
dup_locs = duplicates.to_numpy().nonzero()[0]
if dup_locs.size > 0:
    print(f"Duplicate entries found at rows: {dup_locs}")


