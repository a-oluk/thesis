from functions import *

'''
Initial Parameters defined: 
    - example_txt       = Name of the function
    - function          = the function which is used
    - dim               = Dimension of the function
    - SSN               = Start, stop, and number of points for search space (RoI := Region of Interest)
    - iterations        = Number of iterations for the fpr
    - repetitions       = Number of repetitions for the gpr
    - init_data_size    = Initial data size
    - test_data_size    = Test data size
    - alpha             = Alpha value for calculations
'''

example_txt = "Example 1.1"
function = function_1dim
dim = 1

SSN = -10, 10, 300

iterations = 20
repetitions = 3

init_data_size = 3
test_data_size = 30

alpha = 0.7

'''
Kernel Parameters:
    - rbf_lengthscale     = RBF kernel lengthscale parameter
    - rbf_variance        = RBF kernel variance parameter
    - perio_lengthscale   = Periodic kernel lengthscale parameter
    - perio_periodicity   = Periodic kernel periodicity parameter
    - lin_slope           = Linear kernel slope parameter
    - lin_intercept       = Linear kernel intercept parameter
'''
def init_kernel_params():
    rbf_lengthscale = 1.0
    rbf_variance = 1.0
    perio_lengthscale = 1.0
    perio_periodicity = 1.0
    lin_slope =  1
    lin_intercept = 0
    return rbf_lengthscale, rbf_variance, perio_lengthscale, perio_periodicity, lin_slope, lin_intercept


rbf_lengthscale, rbf_variance, perio_lengthscale, perio_periodicity, lin_slope, lin_intercept = init_kernel_params()

'''
Example Parameters 
'''
def get_params(var_example):
    if var_example == 1.1:
        example_txt = "Example 1.1"
        kernel = rbf_kernel
        function = function_1dim
        SSN = (-10, 10, 100)
        dim = 1
        init_data_size = 5
        repetitions = 1
        iterations = 30
        test_data_size = 30
        alpha = 0.5
    elif var_example == 1.2:
        example_txt = "Example 1.2"
        kernel = rbf_kernel
        function = function_1dim_1
        SSN = (-10, 10, 500)
        dim = 1
        init_data_size = 5
        repetitions = 2
        iterations = 30
        test_data_size = 30
        alpha = 0.5
    elif var_example == 1.3:
        example_txt = "Example 1.3"
        kernel = rbf_kernel
        function = function_1dim_2
        SSN = (-10, 10, 100)
        dim = 1
        init_data_size = 3
        repetitions = 1
        iterations = 15
        test_data_size = 30
        alpha = 0.5
    elif var_example == 1.4:
        example_txt = "Example 1.4"
        kernel = rbf_kernel
        function = function_1dim_3
        SSN = (0, 20, 500)
        dim = 1
        init_data_size = 3
        repetitions = 1
        iterations = 25
        test_data_size = 30
        alpha = 0.5

    elif var_example == 2.1:
        example_txt = "Example 2.1"
        kernel = rbf_kernel
        function = function_2dim
        dim = 2
        init_data_size = 10
        repetitions = 1
        iterations = 30
        test_data_size = 30
        alpha = 0.5
        SSN = (-5, 5, 40)
    elif var_example == 3.1:
        example_txt = "Example 3.1"
        kernel = rbf_kernel
        function = function_3dim
        dim = 3
        init_data_size = 20
        repetitions = 2
        iterations = 20
        test_data_size = 30
        alpha = 0.5
        SSN = (-5, 5, 20)

    elif var_example == 4.1:
        example_txt = "Example 4.1"
        kernel = rbf_kernel
        function = function_4dim
        SSN = (-5, 5, 8)
        dim = 4
        init_data_size = 20
        repetitions = 3
        iterations = 80
        test_data_size = 10
        alpha = 0.5
    elif var_example == 999:
        example_txt = "individual"

    else:
        return None
    return example_txt, kernel, function, SSN, dim, iterations, repetitions, init_data_size, test_data_size, alpha

'''
get the function and dimension
'''

def get_function(str):
    if str == "Example 1.1":
        return function_1dim, 1
    elif str == "Example 1.2":
        return function_1dim_1, 1
    elif str == "Example 1.3":
        return function_1dim_2, 1
    elif str == "Example 2.1":
        return function_2dim, 2
    elif str == "Example 3.1":
        return function_3dim, 3
    elif str == "Example 4.1":
        return function_4dim, 4

    # KANN ICH DAS NICHT ÜBER VALUE MACHEN ?
    elif str == "Example 1.4":
        return function_1dim_3, 1
    else:
        None

"""
    Returns the kernel function based on the given kernel variant and associated parameters.

    Input:
        var_kernel (int): Variant of the kernel - get from Radiobutton of gui
        
    Returns:
        function: kernel function
        
    Raises:
        None
    """

def get_kernel(var_kernel):  #, l=rbf_lengthscale, v=rbf_variance, ls=perio_lengthscale, p=perio_periodicity, ):
    if var_kernel == 0:
        return lambda X1, X2: rbf_kernel(X1, X2, l=rbf_lengthscale, v=rbf_variance)
    elif var_kernel == 1:
        return lambda X1, X2: periodic_kernel(X1, X2, ls=perio_lengthscale, p=perio_periodicity)
    elif var_kernel == 2:
        return lambda X1, X2: linear_kernel(X1, X2,beta_0= lin_intercept,beta_1=lin_slope)
    else:
        return None
