# %%
import numpy as np
from numpy.linalg import inv,matrix_rank,det
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Load the California housing dataset
data = load_breast_cancer(as_frame=True) # Load as a DataFrame

X = data.data
y = data.target

print(X.head())

# %% [markdown]
# ### (i) ###

# %%
# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# %% [markdown]
# ### (ii) ###

# %%
# Feature Selection using training data
# Combine into a single DataFrame
df = X_train.copy()
df['target'] = y_train

# Compute correlation with the target
correlations = df.corr()['target'].drop('target')
print("Correlations with target:")
print(correlations.abs().sort_values(ascending=False))

# Filter features with |correlation| > 0.5
filtered_features = correlations[correlations.abs()>0.5].index.tolist()
print("\nFiltered features with absolute correction > 0.5:")
print(filtered_features)

# Drop features highly correlated with each other (|corr| > 0.9)
selected_features = []
cor_matrix = df[filtered_features].corr().abs()
for feature in filtered_features:
    # Check if it's highly correlated with any already selected feature
    if all(cor_matrix.loc[feature, selected] <= 0.9 for selected in selected_features):
        # all(iterable): returns True if all elements of an iterable are True.
        #                all([]) returns True by definition
        selected_features.append(feature)
print("Final Selected features:")
print(selected_features)

# Subset the DataFrame to only include the selected features.
df_train = X_train[selected_features]
df_val = X_val[selected_features]
df_test = X_test[selected_features]

# Print the shapes of the datasets
print(f"Training set shape after feature selection: {df_train.shape}, {y_train.shape}")
print(f"Validation set shape after feature selection: {df_val.shape}, {y_val.shape}")
print(f"Test set shape after feature selection: {df_test.shape}, {y_test.shape}")



# %% [markdown]
# ### (iii) ###

# %%
# check if X^T * X is invertible
def check_inverse_rank(matrix):
    rank = matrix_rank(matrix, tol=1e-12)
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
order = 1
threshold = 0.5
lamda = 0.00001

# Augment 1 to X
Poly = PolynomialFeatures(order)
X_train_poly = Poly.fit_transform(df_train)
X_val_poly = Poly.fit_transform(df_val)
X_test_poly = Poly.fit_transform(df_test)
    
# Fit a linear regression model
if check_inverse_rank(X_train_poly.T @ X_train_poly):
    print("Applying linear regression without regularization")
    w = inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
else:
    print("Applying linear regression with regularization")
    w = inv(X_train_poly.T @ X_train_poly + lamda * np.eye(X_train_poly.shape[1])) @ X_train_poly.T @ y_train

print(f"w: {w}")

# Predict on the training, validation, and test sets
y_train_pred = ((X_train_poly @ w)>threshold).astype(int)
y_val_pred = ((X_val_poly @ w)>threshold).astype(int)
y_test_pred = ((X_test_poly @ w)>threshold).astype(int)
    
# Calculate and print the accuracy for training, validation sets and test sets
acc_train = accuracy_score(y_train, y_train_pred)
acc_val = accuracy_score(y_val, y_val_pred)
acc_test = accuracy_score(y_test, y_test_pred)
print(f"Training acc: {acc_train}, Validation acc: {acc_val}, Test acc: {acc_test}")
cm_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for Test Set:")
print(cm_test)

# %% [markdown]
# ### (iv) ###

# %%
# Compute prediction, cost and gradient based on binary cross entropy
def logistic_cost_gradient(X, w, y, eps=1e-15):
    z = X @ w  # Linear combination
    
    pred_y = 1/(1+np.exp(-z))
    # Clip predictions to prevent log(0)
    pred_y = np.clip(pred_y, eps, 1 - eps)

    N = X.shape[0]  # Number of samples
    cost   = np.mean(- (y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y)))
    gradient = ((pred_y-y) @ X)/ N  # Normalize by number of samples

    return pred_y, cost, gradient

# %%
# Perform binormial logistic regression
def binormial_logistic_regression(P, w, y, lr, num_iters):
    pred_y, cost, gradient = logistic_cost_gradient(P, w, y)
    print('Initial Cost =', cost)
    print('Initial w =', w)
    cost_vec = np.zeros(num_iters+1)
    cost_vec[0] = cost

    for i in range(1, num_iters + 1):
        w -= lr * gradient
        pred_y, cost, gradient = logistic_cost_gradient(P, w, y)
        cost_vec[i] = cost
        if i % 2000 == 0:
            print(f"Iteration {i}, Cost: {cost}")
    return w, cost_vec, pred_y      

# %%

lr_list = [0.1, 0.01, 0.001]
acc_train_list_Log = [] # store training accuracy for each learning rate
acc_val_list_Log = [] # store validation accuracy for each learning rate
num_iters = 20000
threshold = 0.5
order = 1 # linear features
cost_dict = {}  # a dictionary with key to be the learning rate, and value to be the cost.
max_val_acc = 0
best_lr = 0

# Create polynomial features X to P
Poly = PolynomialFeatures(order)
X_train_poly = Poly.fit_transform(df_train)
X_val_poly = Poly.fit_transform(df_val)
X_test_poly = Poly.fit_transform(df_test)

for lr in lr_list:
    # Initialize weights
    np.random.seed(42)  # set the seed for the random number generator for reproducibility
    w = np.random.randn(X_train_poly.shape[1])# 1D array of random values sampled from the standard normal distribution

    # Fit a logistic regression model
    print(f"Fitting logistic regression for learning rate {lr}...")
    w_opt, cost_vec, y_train_pred = binormial_logistic_regression(X_train_poly, w, y_train, lr, num_iters)
    print(f"w for learning rate {lr}: {w_opt}")

    # Compute training and validation accuracies.
    train_acc = accuracy_score(y_train, (y_train_pred > threshold).astype(int))
    y_val_pred,_,_ = logistic_cost_gradient(X_val_poly, w_opt, y_val)
    val_acc = accuracy_score(y_val, (y_val_pred > threshold).astype(int))
    print(f"Training accuracy for learning rate {lr}: {train_acc}")
    print(f"Validation accuracy for learning rate {lr}: {val_acc} \n")
    acc_val_list_Log.append(val_acc)
    acc_train_list_Log.append(train_acc)

    # store cost
    cost_dict[lr] = cost_vec

    # Check for best validation accuracy for hyperparameter selection
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        best_lr = lr
        y_test_pred,_,_ = logistic_cost_gradient(X_test_poly, w_opt, y_test)
        test_acc = accuracy_score(y_test, (y_test_pred > threshold).astype(int))
        cm_test = confusion_matrix(y_test, (y_test_pred > threshold).astype(int))

print(f"Best learning rate: {best_lr}, Max validation accuracy: {max_val_acc}")
print(f"Test accuracy for the best model: {test_acc}")
print("Confusion Matrix for Test Set:")
print(cm_test)


# %%
print('Training acc: ', str(acc_train_list_Log))
print('Validation acc: ', str(acc_val_list_Log))
# Plot cost values over iterations for each learning rate
plt.figure(0, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
colors = ['r', 'g', 'b']

for i, (lr, cost_vec) in enumerate(cost_dict.items()): #.items(): iterable of key-value pairs
                                                       # enumerate(): Adds an index to each element of an iterable
    plt.plot(np.arange(0, num_iters+1, 1), cost_vec, color=colors[i], label=f'lr={lr}')
             # np.arange(start, stop, step)
plt.legend(loc='upper right', fontsize=15)
plt.xlabel('Iteration Number')
plt.ylabel('Mean Binary Cross Entropy Loss')
plt.xticks(np.arange(0, num_iters+1, 4000))
plt.title('Binomial Logistic Regression')

# %% [markdown]
# # Using sklearn's Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(df_train, y_train)
print("model coefficients:", model.coef_)
print("model intercept:", model.intercept_)
y_pred_train = model.predict(df_train)
train_acc = accuracy_score(y_train, y_pred_train)
y_pred_val = model.predict(df_val)
val_acc = accuracy_score(y_val, y_pred_val)
y_pred_test = model.predict(df_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Training accuracy using sklearn: {train_acc}")
print(f"Validation accuracy using sklearn: {val_acc}")
print(f"Test accuracy using sklearn: {test_acc}")


