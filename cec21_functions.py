from ctypes import CDLL, POINTER, c_int, c_double
import os
import platform
import numpy as np

if platform.system() == "Windows":
    lib = 'trainer/cec17_test_func_win.so'
elif platform.system() == "Linux":
    lib = 'cec21_test_func.so'
else:
    lib = 'Something wrong'

def cec21_test_func(x, f, nx, mx, func_num,
                    dll_path=CDLL(os.path.abspath(lib))):

    if func_num == 31:
        f[0] = sum([value ** 2 for value in x])
        return

    # if func_num == 32:
    #     f[0] = sum(
    #         100.0 *
    #         np.power(
    #             (np.subtract(x[1:], np.power(x[:-1], 2))), 2)
    #         + np.power((np.subtract(1, x[:-1])), 2))
    #     return

    functions = dll_path
    x_pointer_type = POINTER(c_double * nx)
    f_pointer_type = POINTER(c_double * mx)
    nx_type = c_int
    mx_type = c_int
    func_num_type = c_int
    functions.cec21_test_func.argtypes = [x_pointer_type, f_pointer_type,
                                          nx_type, mx_type, func_num_type] 
    functions.cec21_test_func.restype = None
    x_ctype = (c_double * nx)()
    for i, value in enumerate(x):
        x_ctype[i] = value
    f_ctype = (c_double * mx)()
    for i in range(mx):
        f_ctype[i] = 0
    functions.cec21_test_func(x_pointer_type(x_ctype), f_pointer_type(f_ctype),
                              nx, mx, func_num)
    for i in range(len(f)):
        f[i] = f_ctype[i]
