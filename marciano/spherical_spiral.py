import numpy as np
import matplotlib.pyplot as plt


def spherical_spiral_from_north_pole(tangential_arc_length, radial_change_per_revolution):
    """
    Generator function for a spiral in spherical coordinates centering on the north pole.

    :param tangential_arc_length: Desired arc length to move tangentially for each step on a unit sphere.
    :param radial_change_per_revolution: radians to move radially away from the start point for every 2 pi rotated tangentially.

    :return: successive values of theta and phi.
    """

    theta, phi = tangential_arc_length * 2, 0
    yield theta, phi  # Yield the initial position

    # Provide an initial step in theta to move away from the pole
    # theta += tangential_arc_length * 1.2
    # yield theta, phi

    while True:
        # Compute the change in phi to achieve the desired arc length, based on current theta
        if np.sin(theta) != 0:
            delta_phi = tangential_arc_length / np.sin(theta)
        else:
            delta_phi = 0  # At poles, no movement in phi

        phi += delta_phi
        phi %= 2 * np.pi  # Handle wrapping around in phi

        # Update theta based on accumulated change in phi
        theta += (delta_phi / (2 * np.pi)) * radial_change_per_revolution
        theta = np.clip(theta, 0, np.pi)  # Clip theta to remain in [0, pi]

        yield theta, phi


def spherical_spiral(tangential_arc_length, radial_change_per_revolution, start_theta=0, start_phi=0):
    """
    Generator function for a spiral in spherical coordinates centering on a given point.

    :param tangential_arc_length: Desired arc length to move tangentially for each step on a unit sphere.
    :param radial_change_per_revolution: Radians to move radially away from the start point for every 2 pi rotated tangentially.
    :param start_theta: the starting co-latitude.
    :param start_phi: the starting longitude.

    :return: successive values of theta and phi.
    """

    for theta, phi in spherical_spiral_from_north_pole(tangential_arc_length, radial_change_per_revolution):
        yield rotate_point(start_theta, start_phi, theta, phi)


def rotate_point(theta_rot, phi_rot, theta, phi):
    """
    Rotate the point (theta, phi) such that the North Pole (0, 0) goes to (theta_rot, phi_rot).

    Parameters:
    - theta_rot, phi_rot: The coordinates of the point that you want the North Pole to be rotated to.
    - theta, phi: The coordinates of the point that you want to find the new position of after rotation.

    Returns:
    - The rotated coordinates (theta_new, phi_new) of the point (theta, phi).
    """
    if theta_rot == phi_rot == 0:
        return theta, phi
    # Convert to Cartesian coordinates
    x_rot, y_rot, z_rot = sph2cart(1, theta_rot, phi_rot)
    x, y, z = sph2cart(1, theta, phi)

    # Calculate the rotation axis, which is perpendicular to the vector (x_rot, y_rot, z_rot)
    axis = np.array([-y_rot, x_rot, 0])
    axis = axis / np.linalg.norm(axis)

    # Calculate the rotation angle
    angle = -theta_rot

    # Compute the rotation matrix
    R = rotation_matrix(axis, angle)

    # Apply the rotation to the point (x, y, z)
    x_new, y_new, z_new = np.dot(R, [x, y, z])

    # Convert back to spherical coordinates
    r, theta_new, phi_new = cart2sph(x_new, y_new, z_new)

    return theta_new, phi_new


# Helper functions
def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def rotation_matrix(axis, angle):
    """
    Compute the rotation matrix for rotating about a given axis by a given angle.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


if __name__ == '__main__':
    # Generate a set of points using the spiral generator
    N = 2000  # Number of points
    spiral = spherical_spiral(0.02, 0.2, start_theta=np.pi/2, start_phi=np.pi/3)
    points = list(next(spiral) for _ in range(N))

    # Convert points to Cartesian coordinates for plotting
    x, y, z = zip(*[sph2cart(1, theta, phi) for theta, phi in points])

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=np.arange(N), cmap='viridis', marker='o')  # Color points by order for clarity
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
