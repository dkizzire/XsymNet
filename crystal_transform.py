import numpy as np

def lattice_parameters_to_vectors(a, b, c, alpha, beta, gamma):
    """
    Convert lattice parameters to lattice vectors.

    Parameters:
        a, b, c (float): Lattice constants.
        alpha, beta, gamma (float): Angles in degrees between lattice vectors.

    Returns:
        ndarray: 3x3 array of lattice vectors.
    """
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate the components of the lattice vectors
    v1 = [a, 0, 0]
    v2 = [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0]
    v3_x = c * np.cos(beta_rad)
    v3_y = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    v3_z = np.sqrt(c**2 - v3_x**2 - v3_y**2)
    v3 = [v3_x, v3_y, v3_z]

    return np.array([v1, v2, v3])

def calculate_transformation_matrix(params1, params2):
    """
    Calculate the transformation matrix between two sets of lattice parameters.

    Parameters:
        params1 (tuple): Lattice parameters (a, b, c, alpha, beta, gamma) of the first structure.
        params2 (tuple): Lattice parameters (a, b, c, alpha, beta, gamma) of the second structure.

    Returns:
        transformation_matrix (ndarray): 3x3 transformation matrix from lattice1 to lattice2.
    """
    # Convert lattice parameters to lattice vectors
    lattice1 = lattice_parameters_to_vectors(*params1)
    lattice2 = lattice_parameters_to_vectors(*params2)

    # Calculate the inverse of lattice1
    lattice1_inv = np.linalg.inv(lattice1)

    # Compute the transformation matrix
    transformation_matrix = np.dot(lattice2, lattice1_inv)

    return transformation_matrix

# Example usage
if __name__ == "__main__":
    # Define lattice parameters for the two structures
    # bleh = 3.7335 #* gets a,b correct but underestimates c
    # bleh = 3.923  #* gets c correct but overestimates a,b
    # bleh = 3.82675  #* average between previous 2
    # params1 = (bleh, bleh, bleh, 90, 90, 90)  # a, b, c, alpha, beta, gamma
    # params2 = (5.28, 5.28, 7.84, 90, 90, 90)

    params1 = (5.2970, 5.2970, 3.9294, 90, 90, 90)  # a, b, c, alpha, beta, gamma
    params2 = (5.2772, 5.2772, 7.8486, 90, 90, 90)

    # Calculate the transformation matrix
    transformation_matrix = calculate_transformation_matrix(params1, params2)

    print("Transformation Matrix:")
    print(transformation_matrix)
