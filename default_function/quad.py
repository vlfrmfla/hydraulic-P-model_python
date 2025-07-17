
import numpy as np
import warnings

def quadp(A, B, C):
    """Returns the larger root of the quadratic equation A*x^2 + B*x + C = 0"""
    if any(np.isnan([A, B, C])):
        return np.nan
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        warnings.warn("IMAGINARY ROOTS IN QUADRATIC")
        return 0
    if A == 0:
        if B == 0:
            return 0
        else:
            return -C / B
    return (-B + np.sqrt(discriminant)) / (2*A)

def quadm(A, B, C):
    """Returns the smaller root of the quadratic equation A*x^2 + B*x + C = 0"""
    if any(np.isnan([A, B, C])):
        return np.nan
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        warnings.warn("IMAGINARY ROOTS IN QUADRATIC")
        return 0
    if A == 0:
        if B == 0:
            return 0
        else:
            return -C / B
    return (-B - np.sqrt(discriminant)) / (2*A)
