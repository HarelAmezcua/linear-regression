import numpy as np
import matplotlib.pyplot as plt

def least_squares(X,Y):    
    """
    Perform simple linear regression using the least squares method.
    Parameters:
    X (numpy array): Independent variable values.
    Y (numpy array): Dependent variable values.
    Returns:
    list: Coefficients [w0, w1] of the linear regression model, where w0 is the intercept and w1 is the slope.
    """

    # Calculate the mean of X and Y
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    # Multiply X and Y element-wise
    xy = np.multiply(X, Y)
    xx = np.multiply(X, X)

    # Calculate the sum of X, Y, X*Y, X*X
    sum_x = np.sum(X)    
    sum_xy = np.sum(xy)
    sum_xx = np.sum(xx)

    # Calculate the fraction elements
    upper_element = sum_xy - mean_y * sum_x
    lower_element = sum_xx - mean_x * sum_x

    # Calculate the slope and intercept
    w1 = upper_element / lower_element
    w0 = mean_y - w1 * mean_x
    
    w = [w0, w1]
    w = np.array(w).reshape(-1,1)
    return w


def gradient_descent(indep_var,output,learning_rate=0.001,epochs=10000):
    """
    Perform simple linear regression using the gradient descent method.
    Parameters:
    X (numpy array): Independent variable values.
    Y (numpy array): Dependent variable values.
    Returns:
    list: Coefficients [w0, w1] of the linear regression model, where w0 is the intercept and w1 is the slope.
    """

    n = indep_var.shape[0] # number of samples
    m = indep_var.shape[1] # number of the independent variable

    # Initialize the weights
    w = np.random.rand(m+1,1)

    Y = output.copy()
    X = np.c_[np.ones(n), indep_var]
     

    for i in range(epochs):
        w = w + learning_rate * (1/n) * np.dot(X.T, Y - np.dot(X, w))
    return w 


def pseudoinverse_method(indep_var, output):
    """
    Perform simple linear regression using the pseudoinverse method.
    Parameters:
    indep_var (numpy array): Independent variable values.
    output (numpy array): Dependent variable values.
    Returns:
    list: Coefficients [w0, w1] of the linear regression model, where w0 is the intercept and w1 is the slope
    """
    
    n = indep_var.shape[0] # number of samples
    m = indep_var.shape[1] # number of the independent variable

    # Initialize the weights
    w = np.random.rand(m+1,1)

    # Add a column of ones to the independent variable matrix
    Y = output.copy()
    X = np.c_[np.ones(n), indep_var]

    w = np.dot(np.linalg.pinv(X), Y)
    return w

def multidim_vectorized(x,y):
    """
    Perform simple linear regression using the pseudoinverse method.
    Parameters:
    indep_var (numpy array): Independent variable values.
    output (numpy array): Dependent variable values.
    Returns:
    list: Coefficients [w0, w1] of the linear regression model, where w0 is the intercept and w1 is the slope
    """
    
    n = x.shape[0] # number of samples
    m = y.shape[1] # number of the independent variable

    # Initialize the weights
    w = np.random.rand(m+1,1)

    # Add a column of ones to the independent variable matrix
    Y = y.copy()
    X = np.c_[np.ones(n), x]

    X_transpose = np.transpose(X)

    w = np.linalg.inv(X_transpose @ X) @ X_transpose @ Y
    return w

def adaline(x,y):
    """
    Perform simple linear regression using the pseudoinverse method.
    Parameters:
    indep_var (numpy array): Independent variable values.
    output (numpy array): Dependent variable values.
    Returns:
    list: Coefficients [w0, w1] of the linear regression model, where w0 is the intercept and w1 is the slope
    """
    
    n = x.shape[0] # number of samples
    m = y.shape[1] # number of the independent variable

    # Initialize the weights
    w = np.random.rand(m+1,1)

    eta = 0.0001
    
    iterations = 10000
    for i in range(iterations):
        for j in range(n):
            x_j = np.concatenate(([1], x[j].flatten())).reshape(-1,1)
            y_pred = np.dot(np.transpose(w), x_j)
            error = y[j] - y_pred
            w = w + eta * error * x_j

    return w


def r2_score(x,y,w):
    n = x.shape[0] # number of samples
    Y = y.copy()
    X = np.c_[np.ones(n), x]
    y_pred = np.dot(X, w)
    y_pred = y_pred.reshape(-1,1)
    y_mean = np.mean(Y)
    upper_element = np.sum((Y - y_pred)**2)
    lower_element = np.sum((Y - y_mean)**2)
    r2 = 1 - upper_element / lower_element
    return r2


def plot_linear_regression(X,Y,w):
    """
    Create a scatter plot of the data and the linear regression line.
    Parameters:
    X (numpy array): Independent variable values.
    Y (numpy array): Dependent variable values.
    w (list): Coefficients [w0, w1] of the linear regression model, where w0 is the intercept and w1 is the slope.
    Returns:
    matplotlib.figure.Figure: Figure object representing the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Create a scatter plot
    ax.scatter(X, Y)

    # Create the regression line
    x_values = X
    y_values = w[0] + w[1] * x_values
    ax.plot(x_values, y_values, color='red')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linear Regression')
    ax.grid(True)    
    return fig