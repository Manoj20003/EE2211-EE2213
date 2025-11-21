import numpy as np
X = np.array([1,0.8,2,2.5,3,4,4.2,6,6.3,7,8,8.2,9]).T  # Feature values for the samples
Y = np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5]).T  #an example. The numbers are arranged from small to big. It's split into two nodes between 5.2 to 8.9.
print('Y.shape is ',Y.shape)
length_Y=Y.shape[0]
threshold = 5.0    # <-- change this to whatever you want
# -----------------------------------------
left_mask  = X < threshold
right_mask = X >= threshold

# Split Y according to X comparison
Y2 = Y[left_mask]
Y1 = Y[right_mask]
print(f"Y1 is {Y1}\n")
y_mean=np.mean(Y, axis=0)
print(f"y_mean is {y_mean}\n")
y1_mean=np.mean(Y1, axis=0)
print(f"y1_mean is {y1_mean}\n")
y2_mean=np.mean(Y2, axis=0)
print(f"y2_mean is {y2_mean}\n")

y_correct_root=np.array(y_mean*np.ones((len(Y),1)))
print('y_correct_root',y_correct_root)
Y_correct_splitted=np.vstack((y1_mean*np.ones((len(Y1),1)),y2_mean*np.ones((len(Y2),1))))
print("Y_correct_spliitted_root",Y_correct_splitted)


y_difference_square = np.square(Y - y_correct_root)
sum_of_square = sum(y_difference_square)
mean_squared_error = sum_of_square / Y.shape[0]
print("root square error is for root", sum_of_square)
print("root MEAN square error is", mean_squared_error, "\n")


y_difference_square = np.square(Y - Y_correct_splitted)
sum_of_square = sum(y_difference_square)
mean_squared_error = sum_of_square / Y.shape[0]
print("split square error is", sum_of_square)
print("split MEAN square error is", mean_squared_error, "\n")


if len(Y1) > 0:
    left_mse = np.mean((Y1 - y1_mean)**2)
else:
    left_mse = 0  # no samples → no error
print("Left MSE =", left_mse)

# -----------------------------------------
# ⭐ MSE OF RIGHT SPLIT (Y>=X)
# -----------------------------------------
if len(Y2) > 0:
    right_mse = np.mean((Y2 - y2_mean)**2)
else:
    right_mse = 0
print("Right MSE =", right_mse)

# -----------------------------------------
# ⭐ OVERALL WEIGHTED MSE (the real split score)
# -----------------------------------------
N = len(Y)
overall_mse = (len(Y1) * left_mse + len(Y2) * right_mse) / N
print("\nOverall Split MSE =", overall_mse)

root_mse = np.mean((Y - y_mean)**2)
print("Root MSE =", root_mse)