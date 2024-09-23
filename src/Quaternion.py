import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

        self.normalized = False
        self.inverse = None

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

    def inverse(self):
        if self.inverse is None:
            self.inverse = self.conjugate() / self.norm()
        return self.inverse

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
        
        if len(axis) == 4:
            axis = axis[:3]

        sin_half_angle = np.sin(half_angle)
        axis_norm = np.linalg.norm(axis)
        # print (f"sin_half_angle: {sin_half_angle}")

        super().__init__(np.cos(half_angle), * (axis * sin_half_angle / axis_norm))

        self.normalized = True
    
    def rotate(self, v):
        q = Quaternion(0, *v)
        lhs = self
        if not self.normalized:
            rhs = self.inverse()
        else:
            rhs = self.conjugate()

        return (lhs * q * rhs).imag()