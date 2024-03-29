#Python code to perform least square fitting for equations of the forms y = bx+a,y = ax^b and y = ae^(bx).
import numpy as np 
def lsqf():
        # Asking user for the total number of elements, ensuring it's greater than 2
    while True:
        try:
            print("---------------------------------------")
            n = int(input("Enter the total number of elements : "))
            if n>2:
                break # Break out of the loop if input is valid
            else:
                print("Please! Enter a number greater than 2")
        except ValueError:
            print("Error Invalid input.")
    try:
        # Prompting user to choose the type of equation
        print("---------------------------------------------------------")
        print("Type 1 : y = bx+a\nType 2 : y = ax^b\nType 3 : y = ae^(bx) ")
        print("---------------------------------------------------------")
        while True:
            eq_type = int(input("Choose the type of equation(1,2 or 3) : "))
            if eq_type in [1,2,3]:
                break # Break out of the loop if input is valid
            else:
                print("Please! Enter a valid input. ") 

        # Getting input elements for x and y
        print("---------------------------------------------------------")
        print("Enter the elements of x .")
        print("---------------------------------------------------------")
        array1 = [float(input(f"Enter the element {i+1} of x ")) for i in range(n)]
        print("---------------------------------------------------------")
        print("Enter the elements of y .")
        print("---------------------------------------------------------")
        array2 = [float(input(f"Enter the element {i+1} of y ")) for i in range(n)]
        print("---------------------------------------------------------")

        # Preparing log arrays for further computation
        array3 = np.log(array1)
        array4 = np.log(array2)

        # Computing sigma values
        sigma_x = np.sum(array1)
        sigma_y = np.sum(array2)
        sigma_lx = np.sum(array3)
        sigma_ly = np.sum(array4)
        sigma_xy = np.sum(np.multiply(array1,array2))
        sigma_xsq = np.sum(np.square(array1))
        sigma_lxsq = np.sum(np.square(array3))
        sigma_lxly = np.sum(np.multiply(array3,array4))
        sigma_xly = np.sum(np.multiply(array1,array4))

        # Performing calculations based on the chosen equation type
        if eq_type ==1:
            a = (sigma_xsq*sigma_y-sigma_x*sigma_xy)/(n*sigma_xsq-sigma_x**2)
            b = (n*sigma_xy-sigma_x*sigma_y)/(n*sigma_xsq-sigma_x**2)
            return a,b
        elif eq_type ==2:
            a = np.exp((sigma_lxsq*sigma_ly-sigma_lx*sigma_lxly)/(n*sigma_lxsq-sigma_lx**2))   
            b = (n*sigma_lxly-sigma_lx*sigma_ly)/(n*sigma_lxsq-sigma_lx**2)         
            return a,b
        elif eq_type==3:
            a = np.exp((sigma_xsq*sigma_ly-sigma_x*sigma_xly)/(n*sigma_xsq-sigma_x**2))
            b = (n*sigma_xly-sigma_x*sigma_ly)/(n*sigma_xsq-sigma_x**2)  
            return a,b
        else:
            print("Error! Invalid input.")
            return None,None
    except ValueError:
        print("Error! Please enter a valid integer. Try Again!!!")

# Calling the function and printing the results if they are valid
result = lsqf()
if result is not None:
    a,b = result
    print(a,b)
print("---------------------------------------")
print("This code is written by Ram(2023PHY1034)")
print("---------------------------------------")
