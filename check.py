# #python code for least square fitting for the equation in the form of y = ax^b or logy = loga + b*logx 
"""
import numpy as np
# #Creating data entry function to take datapoints from the user.
def lsqf():
    n = int(input( "Enter the total number of elements:"))
    try :
        print("Enter the elements for x")
        array1 = [np.log(float(input(f"Enter element {i+1} of array1 : "))) for i in range(n)]
        print("Enter the elements for y")
        array2 = [np.log(float(input(f"Enter element {i+1} of array2 : "))) for i in range(n)]
        sigma_x = np.sum(array1)
        sigma_y = np.sum(array2)
        sigma_xy = np.sum(np.multiply(array1,array2))
        sigma_xsq = np.sum(np.square(array1))
        m = (n*sigma_xy - sigma_x*sigma_y)/(n*sigma_xsq - sigma_x**2)
        c = np.exp((sigma_xsq*sigma_y - sigma_x*sigma_xy)/(n*sigma_xsq - sigma_x**2))
        return m,c
    except ValueError :
        print("Error! Please Enter numerical value")
        return None,None

m,c = lsqf()
print(f"Slope is {m} and intercept is {c}")
"""
# #python code for least square fitting for the equation in the form of y = ae^(bx) or logy = loga + bx 

# import numpy as np
# # #Creating data entry function to take datapoints from the user.
# def lsqf():
#     n = int(input( "Enter the total number of elements:"))
#     try :
#         print("Enter the elements for x")
#         array1 = [float(input(f"Enter element {i+1} of array1 : ")) for i in range(n)]
#         print("Enter the elements for y")
#         array2 = [np.log(float(input(f"Enter element {i+1} of array2 : "))) for i in range(n)]
#         sigma_x = np.sum(array1)
#         sigma_y = np.sum(array2)
#         sigma_xy = np.sum(np.multiply(array1,array2))
#         sigma_xsq = np.sum(np.square(array1))
#         m = (n*sigma_xy - sigma_x*sigma_y)/(n*sigma_xsq - sigma_x**2)
#         c = np.exp((sigma_xsq*sigma_y - sigma_x*sigma_xy)/(n*sigma_xsq - sigma_x**2))
#         return m,c
#     except ValueError :
#         print("Error! Please Enter numerical value")
#         return None,None

# m,c = lsqf()
# print(f"Slope is {m} and intercept is {c}")
######################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt 
# def lsqf():
#     while True:
#         try:
#             n = int(input("Enter the total number of elements : "))
#             if n>2:
#                 break
#             else:
#                 print("Please! Enter a number greater than 2")
#         except ValueError:
#             print("Error Invalid input.")
#     try:
#         print("Type 1 : y = bx+a\nType 2 : y = ax^b\nType 3 : y = ae^(bx) ")
#         while True:
#             eq_type = int(input("Choose the type of equation(1,2 or 3) : "))
#             if eq_type in [1,2,3]:
#                 break
#             else:
#                 print("Please! Enter a valid input. ") 
#         print("Enter the elements of x .")
#         array1 = [float(input(f"Enter the element {i+1} of x ")) for i in range(n)]
#         print("Enter the elements of y .")
#         array2 = [float(input(f"Enter the element {i+1} of y ")) for i in range(n)]
#         array3 = np.log(array1)
#         array4 = np.log(array2)
#         sigma_x = np.sum(array1)
#         sigma_y = np.sum(array2)
#         sigma_lx = np.sum(array3)
#         sigma_ly = np.sum(array4)
#         sigma_xy = np.sum(np.multiply(array1,array2))
#         sigma_xsq = np.sum(np.square(array1))
#         sigma_lxsq = np.sum(np.square(array3))
#         sigma_lxly = np.sum(np.multiply(array3,array4))
#         sigma_xly = np.sum(np.multiply(array1,array4))


#         if eq_type ==1:
#             a = (sigma_xsq*sigma_y-sigma_x*sigma_xy)/(n*sigma_xsq-sigma_x**2)
#             b = (n*sigma_xy-sigma_x*sigma_y)/(n*sigma_xsq-sigma_x**2)
#             return a,b
#         elif eq_type ==2:
#             a = np.exp((sigma_lxsq*sigma_ly-sigma_lx*sigma_lxly)/(n*sigma_lxsq-sigma_lx**2))   
#             b = (n*sigma_lxly-sigma_lx*sigma_ly)/(n*sigma_lxsq-sigma_lx**2)         
#             return a,b
#         elif eq_type==3:
#             a = np.exp((sigma_xsq*sigma_ly-sigma_x*sigma_xly)/(n*sigma_xsq-sigma_x**2))
#             b = (n*sigma_xly-sigma_x*sigma_ly)/(n*sigma_xsq-sigma_x**2)  
#             return a,b
#         else:
#             print("Error! Invalid input.")
#             return None,None
#     except ValueError:
#         print("Error! Please enter a valid integer. Try Again!!!")
# result = lsqf()
# if result is not None:
#     a,b = result
#     print(a,b)
###############################################################################
# import math

# def gamma(n):
#     if n == 0:
#         return 1
#     else:
#         return n * gamma(n - 1)

# # Example usage to compute factorial (Gamma(n + 1)) using gamma function
# n = 3
# print("Factorial({}) = {}".format(n, gamma(n)))

###############################################################################################################################
# import numpy
# def fact(i):
#     if i<0:
#         raise ValueError("Factorial fuction is not defined for negative values")
#     elif i == 0 or i==1:
#         return 1
#     else:
#         return i*fact(i-1)
# print(fact(6))    

# import math

# def factorial(n):
#     if n < 0:
#         return float('nan')  # Return NaN for negative inputs
#     elif n == 0:
#         return 1
#     elif isinstance(n, float) and n.is_integer():  # Check if n is a float and an integer
#         result = 1
#         for i in range(1, int(n) + 1):
#             result *= i
#         return result
#     else:
#         return math.gamma(n + 1)  # Use gamma function for non-integer inputs

# def gamma_function(z):
#     try:
#         return math.gamma(z)
#     except ValueError:
#         return float('nan')  # Return NaN if gamma function is not defined for the input

# # Example usage
# print("Factorial(5):", factorial(5))   # Output: 120
# print("Gamma(5):", gamma_function(5))  # Output: 24.0

# print("Factorial(3.5):", factorial(3.5))   # Output: 11.631728396567446
# print("Gamma(3.5):", gamma_function(3.5))  # Output: 3.323350970447842

# print("Factorial(-2):", factorial(-2))   # Output: nan
# print("Gamma(-2):", gamma_function(-2))  # Output: nan

# def gamma_recursive(n):
#     if n == 0:
#         return 1
#     else:
#         return n * gamma_recursive(n - 1)

# def factorial_recursive(n):
#     if n == 0:
#         return 1
#     else:
#         return (n ) * factorial_recursive(n - 1)

# # Test the functions
# n = 5
# print("Gamma({}) =".format(n), gamma_recursive(n))
# print("Factorial({}) =".format(n), factorial_recursive(n))

# import math

# def custom_factorial(n):
#     if n < 0:
#         return 1
#     else:
#         return math.factorial(n)

# def legendre_polynomial(n, x):
#     result = 0
#     for k in range(n + 1):
#         coef = (-1) ** k * custom_factorial(2 * n - 2 * k) / (2 ** n * custom_factorial(k) * custom_factorial(n - k) * custom_factorial(n - 2 * k))
#         result += coef * x ** (n - 2 * k)
#     return result

# # Example usage
# x = 0.5
# n = 3
# print(f"The Legendre polynomial P_{n}({x}) =", legendre_polynomial(n, x))

####################################################################################################
    
# n = int(input("Enter a value for n : "))
# m = coeff(n)
# print(m)
# def fact(a):
#     if a==0:
#         return 1
#     else :
#         return (a)*fact(a-1)
# n =2
# m=1
# coefficients = []
# for i in range(m):
#         coef = ((((-1)**m )*fact(2*n-2*i))/((2**n)*fact(n-i)*fact(n-2*i)))
#         coefficients.append(coef)
# print(coefficients)
# def fact(a):
#     if a==0:
#         return 1
#     else :
#         return (a)*fact(a-1)
# n=4
# i=2
# coef = ((fact(2*n-2*i))/((2**n)*fact(n-i)*fact(n-2*i)*fact(i)))
# a= fact(2*n-2*i)
# b = 2**n
# c= fact(n-i)
# d= fact(n-2*i)
# e = fact(i)
# print(coef)
# print(a,b,c,d,e)

# import matplotlib.pyplot as plt
# import numpy as np

# # Define range for x values
# x_values = np.linspace(-10, 10, 100)

# # Calculate y values for y = x
# y_values_x = x_values

# # Calculate y values for y = x^2
# y_values_x_squared = x_values ** 2

# # Plot y = x
# plt.plot(x_values, y_values_x, label='y = x')

# # Plot y = x^2
# plt.plot(x_values, y_values_x_squared, label='y = x^2')

# # Add labels and title
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of y = x and y = x^2')

# # Add legend
# plt.legend()

# # Display the plot
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Function to evaluate Legendre polynomial using coefficients
# def legendre_evaluation(x, coefficients):
#     result = 0
#     for i, coef in enumerate(coefficients):
#         result += coef * (x ** (len(coefficients) - 1 - i))
#     return result

# # Load coefficients from file
# coefficients = np.loadtxt("legendre_coefficients.dat")

# # Generate data points
# num_points = 100
# x = np.linspace(-1, 1, num_points)

# # Calculate Legendre polynomials at each data point using the function
# legendre_values = [legendre_evaluation(x, coefficients) for x in x]

# # Plot the Legendre polynomial
# plt.plot(x, legendre_values, label="Legendre Polynomial")
# plt.xlabel("x")
# plt.ylabel("P_n(x)")
# plt.title("Legendre Polynomial")
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def fact(a):
#     if a == 0:
#         return 1
#     else:
#         return a * fact(a - 1)

# def coeff(n):
#     coefficients = []
#     if n % 2 == 0:
#         m = n // 2
#     else:
#         m = (n - 1) // 2
    
#     for i in range(m + 1):
#         b = i
#         coef = ((((-1) ** b) * fact(2 * n - 2 * i)) / ((2 ** n) * fact(n - i) * fact(n - 2 * i) * fact(i)))
#         coefficients.append(coef)
#     return coefficients

# def legendre(n, x):
#     if n == 0:
#         return 1
#     elif n == 1:
#         return x
#     else:
#         P_nm2 = 1
#         P_nm1 = x
#         for k in range(2, n + 1):
#             coef = (2 * k - 1) / k
#             P_n = coef * x * P_nm1 - (k - 1) / k * P_nm2
#             P_nm2, P_nm1 = P_nm1, P_n
#             print(P_n)
#         return P_n
# def save_coefficients(n, coefficients):
#     filename = f"legendre_coefficients_{n}.dat"
#     np.savetxt(filename, coefficients)
#     print(f"Legendre polynomial coefficients for degree {n} saved in {filename}")

# def plot_legendre(n, coefficients):
#     num_points = 100
#     x = np.linspace(-1, 1, num_points)
#     legendre_values = np.array([legendre(n, x_i) for x_i in x])
#     print(legendre_values)
#     plt.plot(x, legendre_values, label=f"P_{n}(x)")
#     plt.xlabel("x")
#     plt.ylabel(f"P_{n}(x)")
#     plt.title(f"Legendre Polynomial of Degree {n}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# n = int(input("Enter the value for n: "))
# coefficients = coeff(n)
# save_coefficients(n, coefficients)
# plot_legendre(n, coefficients)
# import numpy as np
# import matplotlib.pyplot as plt

# def fact(a):
#     if a==0:
#         return 1
#     else :
#         return (a)*fact(a-1) 
# def legendre_my():
#     n = int(input("Enter the value of n: "))
#     x = np.linspace(-1,1,100)
#     if n % 2 == 0:
#         m = n // 2
#     else:
#         m = (n - 1) // 2
#     for i in range(m+1):
#         legendre_poly = 0
#         legendre_poly += (((((-1) ** i) * fact(2 * n - 2 * i)) / ((2 ** n) * fact(n - i) * fact(n - 2 * i) * fact(i))))*x**(n-2*i)
#         return legendre_poly
# a=legendre_my()
# print(a)
# x = np.linspace(-1,1,100)
# plt.plot(x,a)
# plt.show()
###################################################################################################
#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
#defining factorial function
def fact(a):
    if a == 0:
        return 1
    else:
        return a * fact(a - 1) 
#calculating legendre polynomials
def legendre_my(n_values):
    x = np.linspace(-1, 1, 100)
    legendre_polynomials = []
    for n in n_values:
        if n % 2 == 0:
            m = n // 2
        else:
            m = (n - 1) // 2
        legendre_polynomial = np.zeros_like(x)
        for i in range(m + 1):
            legendre_polynomial += ((((-1) ** i) * fact(2 * n - 2 * i)) / ((2 ** n) * fact(n - i) * fact(n - 2 * i) * fact(i))) * (x ** (n - 2 * i))
        legendre_polynomials.append(legendre_polynomial)
    return legendre_polynomials

n_values = [1, 2, 3, 4, 5]
legendre_polynomials = legendre_my(n_values)

x = np.linspace(-1, 1, 100)
plt.figure(figsize=(10, 6))

for n, legendre_poly in zip(n_values, legendre_polynomials):
    plt.plot(x, legendre_poly, label=f'n={n}')

plt.title("Legendre Polynomials")
plt.xlabel("x")
plt.ylabel("P_n(x)")
plt.legend()
plt.grid(True)
plt.show()




































































































