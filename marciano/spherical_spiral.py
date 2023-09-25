import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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


def precompute_spiral(radial_change_per_revolution, ds=0.01):
    """
    Precompute a spiral's coordinates up to the point that it reaches the south pole.

    Parameters:
    - radial_change_per_revolution: The change in radial distance (theta) for each full revolution around the spiral.
    - ds: The differential arc length for which the change in theta and phi is calculated.

    Returns:
    - A tuple of arrays: (arc_lengths, thetas, phis)
    """

    thetas, phis, arc_lengths = [0], [0], [0]
    theta, phi, traveled_length = 0, 0, 0

    while theta < np.pi:
        # Calculate dphi based on current theta
        dphi = ds / np.sqrt(radial_change_per_revolution ** 2 + (np.sin(theta) ** 2))

        # Calculate dtheta based on radial_change_per_revolution and dphi
        dtheta = radial_change_per_revolution * dphi / (2 * np.pi)

        # Update theta and phi
        theta += dtheta
        phi += dphi

        # Update the traveled length
        traveled_length += ds

        thetas.append(theta)
        phis.append(phi)
        arc_lengths.append(traveled_length)

    return np.array(arc_lengths), np.array(thetas), np.array(phis)


_precomputed_spirals = {}

def position_on_spiral(arc_length, radial_change_per_revolution, start_theta=0, start_phi=0, ds=0.01):
    """
    Given an arc length and a radial change per revolution, compute the position on a spiral
    centered on the North Pole in spherical coordinates. The spiral is constructed such that
    the distance between successive points (in terms of arc length) is approximately equal.

    Parameters:
    - arc_length: The total arc length moved along the spiral.
    - radial_change_per_revolution: The change in radial distance (theta) for each full revolution around the spiral.
    - start_theta: theta of center of the spiral
    - start_phi: phi of center of the spiral
    - ds: The differential arc length for which the change in theta and phi is calculated.

    Returns:
    - The spherical coordinates (theta, phi) of the position on the spiral.
    """
    if (radial_change_per_revolution, ds) not in _precomputed_spirals:
        arc_lengths, thetas, phis = precompute_spiral(radial_change_per_revolution, ds)
        # Use linear interpolation to compute the theta and phi values
        theta_interp = interp1d(arc_lengths, thetas, kind='linear', fill_value="extrapolate")
        phi_interp = interp1d(arc_lengths, phis, kind='linear', fill_value="extrapolate")
        _precomputed_spirals[(radial_change_per_revolution, ds)] = theta_interp, phi_interp
    else:
        theta_interp, phi_interp = _precomputed_spirals[(radial_change_per_revolution, ds)]

    theta, phi = theta_interp(arc_length), phi_interp(arc_length) % (2 * np.pi)
    theta %= 2 * np.pi

    if start_phi == start_theta == 0:
        return theta, phi
    else:
        return rotate_point(start_theta, start_phi, theta, phi)


if __name__ == '__main__':
    # Generate a set of points on a spiral
    N = 2000
    arc_lengths = np.linspace(0, 100, N)
    points = [position_on_spiral(arc_length,0.1, np.pi/4, np.pi/3) for arc_length in arc_lengths]
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
