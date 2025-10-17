# %%
import numpy as np
from numpy.linalg import inv,matrix_rank,det
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing(as_frame=True) # Load as a DataFrame

X = data.data
y = data.target

print(X.head())

# %% [markdown]
# ### (i) ###

# %%
## Split the data into training, validation, and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
# random_state: controls the random shuffling of data before splitting into train and test sets.

# Print the shapes of the datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# %% [markdown]
# ### (ii) ###

# %%
## Feature Selection using Pearson's correlation
# Combine into a single DataFrame
df = X_train.copy()
df['target'] = y_train # add one column named "target" for target output
df

# %%

# Compute correlation with the target
correlations = df.corr()['target'].drop('target')
print("Correlations with target:")
print(correlations.abs().sort_values(ascending=False))

# Get top 2 absolute correlations
top2_features = correlations.abs().sort_values(ascending=False).head(2).index.tolist()
print("Top 2 features with highest absolute correlation to target:")
print(top2_features)

# Subset the DataFrame to only include the top 2 features
df_train = X_train[top2_features]
df_val = X_val[top2_features]
df_test = X_test[top2_features]

# Print the shapes of the datasets
print(f"Training set shape after feature selection: {df_train.shape}, {y_train.shape}")
print(f"Validation set shape after feature selection: {df_val.shape}, {y_val.shape}")
print(f"Test set shape after feature selection: {df_test.shape}, {y_test.shape}")

# %%
# test cell
df.corr()
# a Pd method that computes the pairwise correlation between numeric columns in a DataFrame

# %%
# test cell
df.corr()['target']

# %%
# test cell
df.corr()['target'].drop('target')
# remove entries for target itself

# %% [markdown]
# ### (iii) ###

# %%
# check if X^T * X is invertible
def check_inverse_rank(matrix):
    rank = matrix_rank(matrix, tol=1e-12) # tol: Helps avoid counting very small numbers 
                                          #      due to floating-point errors as non-zero.
    print("matrix rank is : "+ str(rank))
    print("matrix size is : "+ str(matrix.shape))

    if matrix.shape[0] == matrix.shape[1]:
       if rank == matrix.shape[0]:
           print("matrix is invertible")
       else:
           print("matrix is not invertible")
    else:
       print("matrix is not square, hence not invertible")

    return (rank == matrix.shape[0]) and (matrix.shape[0] == matrix.shape[1])

def check_inverse_det(matrix, tol=1e-12):
    deter = det(matrix)
    print("determinant is : " + str(deter))
    if abs(deter) < tol:
        print("matrix is invertible")
    else:
        print("matrix is not invertible")

# %%
# Initialize lists to store mean squared errors for each order
mse_train_list = []
mse_val_list = []
mse_test_list = []
max_order = 6

for order in range(1,max_order + 1):
    print(f"Polynomial regression model for order {order}:")
    # Create polynomial features X to P
    Poly = PolynomialFeatures(order)
    X_train_poly = Poly.fit_transform(df_train)
    X_val_poly = Poly.transform(df_val)
    X_test_poly = Poly.transform(df_test)

    # Fit a linear regression model
    if check_inverse_rank(X_train_poly.T @ X_train_poly):
        print("Applying Polynomial Regression without regularization")
        # learning (working out w)
        w = inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
        print(f"w for order {order}: {w}")
    else:
        print("Applying regularization")
        lambda_reg = 1e-5  # Regularization strength
        w = inv(X_train_poly.T @ X_train_poly + lambda_reg * np.eye(X_train_poly.shape[1])) @ X_train_poly.T @ y_train
        
    # Predict on the training, validation, and test sets
    y_train_pred = X_train_poly @ w
    y_val_pred = X_val_poly @ w
    y_test_pred = X_test_poly @ w

    # Calculate and print the mean squared errors for training, validation sets, test sets.
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"Order {order} - Training MSE: {mse_train}, Validation MSE: {mse_val}, Test MSE: {mse_test} \n")
    mse_train_list.append(mse_train)
    mse_val_list.append(mse_val)
    mse_test_list.append(mse_test)
   


# %%
print('Training MSE: ', str(mse_train_list))
print('Validation MSE: ', str(mse_val_list))

# Plot MSE
plt.figure(1, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.plot(np.linspace(1,max_order,max_order), mse_train_list, label='training MSE')
plt.plot(np.linspace(1,max_order,max_order), mse_val_list, label='validation MSE')
plt.xlabel('order') 
plt.ylabel('MSE')
plt.legend(loc='upper left', fontsize=15)
plt.show()

# %%
# Best polynomial order should be selected according to validation set.
print("MSE for test set is : " + str(mse_test_list[3]))

# %% [markdown]
# ### (iv) ###

# %%
# Regularization
mse_train_list_reg = []
mse_val_list_reg = []
mse_test_list_reg = []
max_order = 6
lamda = 1

for order in range(1,max_order + 1):

    # Create polynomial features X to P
    Poly = PolynomialFeatures(order)
    X_train_poly = Poly.fit_transform(df_train)
    X_val_poly = Poly.transform(df_val)
    X_test_poly = Poly.transform(df_test)

    # learning (work out w)
    reg_L = lamda*np.identity(X_train_poly.shape[1])
    w_reg = inv(X_train_poly.T @ X_train_poly + reg_L) @ X_train_poly.T @ y_train
    print(f"Model coefficients for order {order}: {w_reg}")

    # Predict on the training, validation, and test sets
    y_train_pred = X_train_poly @ w_reg
    y_val_pred = X_val_poly @ w_reg
    y_test_pred = X_test_poly @ w_reg

    # Calculate and print the mean squared error for training, validation and test sets.
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"Order {order} - Training MSE: {mse_train}, Validation MSE: {mse_val}, Test MSE: {mse_test} \n")
    mse_train_list_reg.append(mse_train)
    mse_val_list_reg.append(mse_val)
    mse_test_list_reg.append(mse_test)



# %%
print('====== With Regularization =======')
print('Training MSE: ', str(mse_train_list_reg))
print('Validation MSE: ', str(mse_val_list_reg))

# Plot MSE
plt.figure(1, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.plot(np.linspace(1,max_order,max_order), mse_train_list_reg, label='training MSE')
plt.plot(np.linspace(1,max_order,max_order), mse_val_list_reg, label='validation MSE')
plt.xlabel('order') 
plt.ylabel('MSE')
plt.title('With Regularization lamda=' + str(lamda))
plt.legend(loc='upper left', fontsize=15)
plt.show()

# %%
# Plot MSE together
plt.figure(4, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
plt.plot(np.linspace(1,max_order,max_order), mse_train_list, color = "yellow", label='training MSE without reg')
plt.plot(np.linspace(1,max_order,max_order), mse_val_list, color = "red", label='validation MSE without reg')
plt.plot(np.linspace(1,max_order,max_order), mse_train_list_reg, color = "green", label='training MSE with reg')
plt.plot(np.linspace(1,max_order,max_order), mse_val_list_reg, color = "blue", label='validation MSE with reg')
plt.xlabel('order') 
plt.ylabel('MSE')

plt.legend(loc='upper left', fontsize=15)
plt.show()

# %%
print("MSE for test set is : " + str(mse_test_list[3]))


