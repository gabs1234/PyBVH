import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def real(self):
        return self.w

    def imag(self):
        return np.array([self.x, self.y, self.z])
    
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def cross_product (vec1, vec2):
        x = vec1.y * vec2.z - vec1.z * vec2.y
        y = vec1.z * vec2.x - vec1.x * vec2.z
        z = vec1.x * vec2.y - vec1.y * vec2.x
        return Quaternion(0, x, y, z)

    def dot_product (vec1, vec2):
        dot = vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z
        return Quaternion(dot, 0, 0, 0)
    
    def scalar_product (vec1, scalar):
        return Quaternion(vec1.w * scalar, vec1.x * scalar, vec1.y * scalar, vec1.z * scalar)
    
    def norm(self):
        return np.linalg.norm([self.w, self.x, self.y, self.z])

    def normalize(self):
        norm = self.norm()
        self.x = self.x / norm
        self.y = self.y / norm
        self.z = self.z / norm
        return self

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        
        result = Quaternion(self.w * other.w, 0, 0, 0) \
            + Quaternion.cross_product(self, other) \
            - Quaternion.dot_product(self, other) \
            + Quaternion.scalar_product(self, other.w) \
            + Quaternion.scalar_product(other, self.w)
        return result
    
    def __str__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

class RotationQuaternion(Quaternion):
    def __init__(self, angle, axis):
        self.angle = angle

        half_angle = angle / 2
        axis = np.array(axis)
        sin_half_angle = np.sin(half_angle)
        axis_norm = np.linalg.norm(axis)
        print (f"sin_half_angle: {sin_half_angle}")

        super().__init__(np.cos(half_angle), * (axis * sin_half_angle / axis_norm))
    
    def rotate(self, v):
        q = Quaternion(0, *v)
        return (self * q * self.conjugate()).imag()


def plotVectors(plotter, vectors, colors):
    for i, vector in enumerate(vectors):
        center = np.zeros(3)
        plotter.add_arrows(center, vector, mag=.3, color=colors[i])

def plotVector(plotter, origin, vector, color, mag=.3):
    plotter.add_arrows(origin, vector, mag=mag, color=color)

def rotateBasis(u1, u2, u3, theta, phi):
    # Rotate basis
    rotation = RotationQuaternion(-theta, u1)
    u1 = rotation.rotate(u1)
    u2 = rotation.rotate(u2)
    u3 = rotation.rotate(u3)

    rotation = RotationQuaternion(phi, u3)
    u1 = rotation.rotate(u1)
    u2 = rotation.rotate(u2)
    u3 = rotation.rotate(u3)

    print (type(u1))

    return u1, u2, u3


def plotSurfacePoints(mesh_origin, surface_origin, Nx, Ny, Dx, Dy, r, theta, phi):
    # 3d plot
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid()

    # cloud data
    points = []
    colors = []

    # Plot mesh origin
    points.append(mesh_origin)
    colors.append(0)

    # Mesh basis
    u1 = np.array([1, 0, 0])
    u2 = np.array([0, 1, 0])
    u3 = np.array([0, 0, 1])
    basis = [u1, u2, u3]

    # Plot mesh basis
    for i in range(3):
        plotVector(plotter, mesh_origin, basis[i], '#ff0000')

    # Plot surface origin
    points.append(surface_origin)
    colors.append(.5)
    

    # Surface basis
    s1 = np.array([1, 0, 0])
    s2 = np.array([0, 1, 0])
    s3 = np.array([0, 0, 1])

    # Rotate surface basis
    e1, e2, e3 = rotateBasis(s1, s2, s3, theta, phi)
    delta_x = Dx / Nx
    delta_y = Dy / Ny
    e1 *= delta_x
    e2 *= delta_y
    basis = [e1, e2, e3]

    # Plot surface basis
    for i in range(3):
        if i == 2:
            plotVector(plotter, surface_origin, basis[i], '#ff0000', mag=.3)
        else:
            plotVector(plotter, surface_origin, basis[i], '#00ff11', mag=1)

    for i in range(-Nx // 2, Nx // 2):
        for j in range(-Ny // 2, Ny // 2):
            point = surface_origin + i * e1 + j * e2
            points.append(point)
            colors.append(1)
    
    points = np.array(points)
    colors = np.array(colors)

    plotter.add_points(points, render_points_as_spheres=True, point_size=10, scalars=colors)
    plotter.show()


def exampleRotateBasis():
    # Define reference basis 
    u1 = np.array([1, 0, 0])
    u2 = np.array([0, 1, 0])
    u3 = np.array([0, 0, 1])

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid()

    # Plot reference basis
    plotVectors(plotter, [u1, u2, u3], ['#ff0000', '#00ff11', '#0026ff'])

    # Rotate basis
    theta = np.pi
    phi = np.pi

    u1, u2, u3 = rotateBasis(u1, u2, u3, theta, phi)
    plotVectors(plotter, [u1, u2, u3], ['#ffabab', '#b1ffab', '#abb7ff'])

    plotter.show()

def exampleRotateVector(angle, axis, vector):
    # Rotate vector
    rotation = RotationQuaternion(angle, axis)
    u1 = rotation.rotate(vector)

    return u1

def main():
    # Parameters
    mesh_origin = np.array([0, 0, 0])
    surface_origin = np.array([1, -1, -1])
    Nx = 10
    Ny = 10
    Dx = 1
    Dy = 1
    r = 2
    theta = np.pi / 6
    phi = np.pi / 4

    plotSurfacePoints(mesh_origin, surface_origin, Nx, Ny, Dx, Dy, r, theta, phi)

if __name__ == "__main__":
    main()

    # axis = [0.0, 0.0, 1.0]
    # angle = np.pi
    # vector = [1, 0, 0]
    # rotation = RotationQuaternion(angle, axis)
    # u1 = rotation.rotate(vector)
    # print(u1)