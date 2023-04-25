from functions import *

example_txt = "Example 1.1"
function = function_1dim
dim = 1

SSN = -10, 10, 300

iterations = 20
repetitions = 3

init_data_size = 3
test_data_size = 30

alpha = 0.7


def init_kernel_params():
    rbf_lengthscale = 1.0
    rbf_variance = 1.0
    perio_lengthscale = 1.0
    perio_periodicity = 1.0
    return rbf_lengthscale, rbf_variance, perio_lengthscale, perio_periodicity


rbf_lengthscale, rbf_variance, perio_lengthscale, perio_periodicity = init_kernel_params()


def get_params(var_example):
    if var_example == 0:
        example_txt = "Example 1.1"
        kernel = rbf_kernel
        function = function_1dim
        SSN = (-10, 10, 100)
        dim = 1
        init_data_size = 5
        repetitions = 1
        iterations = 30
        test_size = 30
        alpha = 0.5
    elif var_example == 1:
        example_txt = "Example 1.2"
        kernel = rbf_kernel
        function = function_1dim_1
        SSN = (-10, 10, 500)
        dim = 1
        init_data_size = 5
        repetitions = 2
        iterations = 30
        test_size = 30
        alpha = 0.5
    elif var_example == 2:
        example_txt = "Example 1.3"
        kernel = rbf_kernel
        function = function_1dim_2
        SSN = (-10, 10, 100)
        dim = 1
        init_data_size = 3
        repetitions = 1
        iterations = 15
        test_size = 30
        alpha = 0.5
    elif var_example == 3:
        example_txt = "Example 2.1"
        kernel = rbf_kernel
        function = function_2dim
        dim = 2
        init_data_size = 10
        repetitions = 1
        iterations = 30
        test_size = 30
        alpha = 0.5
        SSN = (-5, 5, 40)
    elif var_example == 4:
        example_txt = "Example 4.1"
        kernel = rbf_kernel
        function = function_4dim
        SSN = (-5, 5, 3)
        dim = 4
        init_data_size = 20
        repetitions = 10
        iterations = 10
        test_size = 10
        alpha = 0.5
    elif var_example == 10:
        example_txt = "individual"

    else:
        return None
    return example_txt, kernel, function, SSN, dim, iterations, repetitions, init_data_size, test_size, alpha


def get_function(str):
    if str == "Example 1.1":
        return function_1dim, 1
    elif str == "Example 1.2":
        return function_1dim_1, 1
    elif str == "Example 1.3":
        return function_1dim_2, 1
    elif str == "Example 2.1":
        return function_2dim, 2
    elif str == "Example 4.1":
        return function_4dim, 4
    else:
        None


def get_kernel(var_kernel, l=rbf_lengthscale, v=rbf_variance, ls=perio_lengthscale, p=perio_periodicity):
    if var_kernel == 0:
        return lambda X1, X2: rbf_kernel(X1, X2, l=rbf_lengthscale, v=rbf_variance)
    elif var_kernel == 1:
        return lambda X1, X2: periodic_kernel(X1, X2, ls=perio_lengthscale, p=perio_periodicity)
    elif var_kernel == 2:
        return lambda X1, X2: linear_kernel(X1, X2)
