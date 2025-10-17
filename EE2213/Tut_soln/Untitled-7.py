# %%
import numpy as np
from numpy.linalg import inv,matrix_rank,det
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Load the California housing dataset
data = load_iris() # Load as a DataFrame

X = data.data
y = data.target
print(X)
print(y)

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
# Convert the target variable to one-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
reshaped = y_train.reshape(len(y_train), 1) # 1D array to 2D array
Ytr_onehot = onehot_encoder.fit_transform(reshaped)

reshaped = y_val.reshape(len(y_val), 1)
Yval_onehot = onehot_encoder.fit_transform(reshaped)

reshaped = y_test.reshape(len(y_test), 1)
Yts_onehot = onehot_encoder.fit_transform(reshaped)

Yts_onehot


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
lamda = 0.00001

# Augment 1 to X
Poly = PolynomialFeatures(order)
X_train_poly = Poly.fit_transform(X_train)
X_val_poly = Poly.fit_transform(X_val)
X_test_poly = Poly.fit_transform(X_test)

# Fit a linear regression model
if check_inverse_rank(X_train_poly.T @ X_train_poly):
    print("Applying linear regression without regularization")
    W = inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ Ytr_onehot
else:
    print("Applying linear regression with regularization")
    W = inv(X_train_poly.T @ X_train_poly + lamda * np.eye(X_train_poly.shape[1])) @ X_train_poly.T @ Ytr_onehot

print(f"W: {W}")
# Predict on the training, validation, and test sets
Ytr_est = X_train_poly @ W # probability
Ytr_class = np.argmax(Ytr_est, axis=1) #class predictions
Yval_est = X_val_poly @ W
Yval_class = np.argmax(Yval_est, axis=1)
Yts_est = X_test_poly @ W
Yts_class = np.argmax(Yts_est, axis=1)

# Calculate and print the accuracy for training, validation sets and test sets
acc_train = accuracy_score(y_train, Ytr_class)
acc_val = accuracy_score(y_val, Yval_class)
acc_test = accuracy_score(y_test, Yts_class)
print(f"Training acc: {acc_train}, Validation acc: {acc_val}, Test acc: {acc_test}")
cm_test = confusion_matrix(y_test, Yts_class)
print("Confusion Matrix for Test Set:")
print(cm_test)

# %% [markdown]
# ### (iv) ###

# %%
# Compute prediction, cost and gradient based on categorical cross entropy
def multi_logistic_cost_gradient(X, W, Y, eps=1e-15):
    z = X @ W
    z_max = np.max(z, axis=-1, keepdims=True)  # for numerical stability
    exp_z = np.exp(z - z_max)
    pred_Y = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # Clip predictions to prevent log(0)
    pred_Y = np.clip(pred_Y, eps, 1 - eps)
    
    N = X.shape[0]  # Number of samples
    cost   = (np.sum(-(Y * np.log(pred_Y))))/N
    gradient = (X.T @ (pred_Y-Y))/N

    return pred_Y, cost, gradient

# %%
# Perform multinormial logistic regression
def multinormial_logistic_regression(P, W, Y, lr, num_iters):
    pred_Y, cost, gradient = multi_logistic_cost_gradient(P, W, Y)
    print('Initial Cost =', cost)
    print('Initial Weights =', W)
    cost_vec = np.zeros(num_iters+1)
    cost_vec[0] = cost

    for i in range(1, num_iters + 1):
        W -= lr * gradient
        pred_Y, cost, gradient = multi_logistic_cost_gradient(P, W, Y)
        cost_vec[i] = cost
        if i % 2000 == 0:
            print(f"Iteration {i}, Cost: {cost}")

    return W, cost_vec, pred_Y       

# %%
lr_list = [0.1, 0.01, 0.001]
acc_train_list_Log = []
acc_val_list_Log = []
num_iters = 20000
order = 1
cost_dict = {}
max_val_acc = 0
best_lr = 0

# Augment 1 to X
Poly = PolynomialFeatures(order)
X_train_poly = Poly.fit_transform(X_train)
X_val_poly = Poly.fit_transform(X_val)
X_test_poly = Poly.fit_transform(X_test)

for lr in lr_list:
    # Initialize weights
    np.random.seed(42)  # For reproducibility
    W = np.random.randn(X_train_poly.shape[1], Ytr_onehot.shape[1])

    # Fit a logistic regression model
    print(f"Fitting logistic regression for learning rate {lr}...")
    W_opt, cost_vec, Ytr_est = multinormial_logistic_regression(X_train_poly, W, Ytr_onehot, lr, num_iters)
    print(f"W for learning rate {lr}: {W_opt}")

    # Compute training, validataion accuracies.
    train_acc = accuracy_score(y_train, np.argmax(Ytr_est, axis=1))
    Yval_est,_,_ = multi_logistic_cost_gradient(X_val_poly, W_opt, Yval_onehot)
    val_acc = accuracy_score(y_val, np.argmax(Yval_est, axis=1))
    print(f"Training accuracy for learning rate {lr}: {train_acc}")
    print(f"Validation accuracy for learning rate {lr}: {val_acc} \n")
    acc_train_list_Log.append(train_acc)
    acc_val_list_Log.append(val_acc)

    # store cost vector for each learning rate
    cost_dict[lr] = cost_vec

    # Check for the best validation accuracy
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        best_lr = lr
        Yts_est,_,_ = multi_logistic_cost_gradient(X_test_poly, W_opt, Yts_onehot)
        test_acc = accuracy_score(y_test, np.argmax(Yts_est, axis=1))
        cm_test = confusion_matrix(y_test, np.argmax(Yts_est, axis=1))

print(f"Best learning rate: {best_lr}, Max validation accuracy: {max_val_acc}")
print(f"Test accuracy for the best learning rate {best_lr}: {test_acc}")
print("Confusion Matrix for Test Set:")
print(cm_test)


# %%
print('Training acc: ', str(acc_train_list_Log))
print('Validation acc: ', str(acc_val_list_Log))
# Plot cost function values over iterations for each learning rate
plt.figure(0, figsize=[9,4.5])
plt.rcParams.update({'font.size': 16})
colors = ['r', 'g', 'b']

for i, (lr, cost_vec) in enumerate(cost_dict.items()):
    plt.plot(np.arange(0, num_iters+1, 1), cost_vec, color=colors[i], label=f'lr={lr}')
plt.legend(loc='upper right', fontsize=15)
plt.xlabel('Iteration Number')
plt.ylabel('Categorical Cross Entropy Cost')
plt.xticks(np.arange(0, num_iters+1, 4000))
plt.title('Multinormial_logistic_regression')

# %% [markdown]
# # Using sklearn's Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("model coefficients:", model.coef_)
print("model intercept:", model.intercept_)
y_pred_train = model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
y_pred_val = model.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)
y_pred_test = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Training accuracy using sklearn: {train_acc}")
print(f"Validation accuracy using sklearn: {val_acc}")
print(f"Test accuracy using sklearn: {test_acc}")


