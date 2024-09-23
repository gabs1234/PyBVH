import numpy as np
from Quaternion import RotationQuaternion as rq

class Basis:
    def __init__(self, u, v, w, origin):
        self.u = u
        self.v = v
        self.w = w
        self.origin = origin
    
    def normalize(self):
        self.u = self.u / np.linalg.norm(self.u)
        self.v = self.v / np.linalg.norm(self.v)
        self.w = self.w / np.linalg.norm(self.w)
        return self

    def translate(self, translation):
        self.origin += translation
    
    def rotate_spherical(self, theta, phi):
        # theta is the angle from the z-axis
        # phi is the angle from the x-axis
        rot1 = rq(-theta, self.u)
        rot2 = rq(phi, self.w)

        self.u = rot1.rotate(self.u)
        self.v = rot1.rotate(self.v)
        self.w = rot1.rotate(self.w)

        self.u = rot2.rotate(self.u)
        self.v = rot2.rotate(self.v)
        self.w = rot2.rotate(self.w)
        return self
    
    def rotate_euler(self, yaw, pitch, roll):
        # yaw is the angle from the z-axis
        # pitch is the angle from the x-axis
        # roll is the angle from the y-axis
        if yaw != 0:
            rot1 = rq(yaw, self.v)
            self.u = rot1.rotate(self.u)
            self.w = rot1.rotate(self.w)
        
        if pitch != 0:
            rot2 = rq(pitch, self.u)
            self.v = rot2.rotate(self.v)
            self.w = rot2.rotate(self.w)
        
        if roll != 0:
            rot3 = rq(roll, self.w)
            self.u = rot3.rotate(self.u)
            self.v = rot3.rotate(self.v)

        return self
    
    def scale(self, scale):
        self.u *= scale
        self.v *= scale
        self.w *= scale
        return self

    def scale_pointwise(self, *scale):
        self.u *= scale[0]
        self.v *= scale[1]
        self.w *= scale[2]
        return self
    
    def getPointInBasis(self, x, y, z):
        return self.origin + x*self.u + y*self.v + z*self.w
    
    def getForwardVector(self):
        return self.w

    def getBaseVectors(self):
        # append u,v,w with a 4th element 0
        u = np.append(self.u, 0).astype(np.float32)
        v = np.append(self.v, 0).astype(np.float32)
        w = np.append(self.w, 0).astype(np.float32)
        return u, v, w
    
    def getOrigin(self):
        return np.append(self.origin, 0).astype(np.float32)
        
class CameraBasis (Basis):
    def __init__(self, r, theta, phi, meshOrigin):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # compute the camera basis
        W = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dtype=np.float32)
        U = np.array([cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta], dtype=np.float32)
        V = np.array([-sin_phi, cos_phi, 0], dtype=np.float32)

        if len(meshOrigin) >= 4:
            meshOrigin = meshOrigin[:3]

        origin = meshOrigin + r * W

        W = -W
        U = -U
        V = -V
        
        super().__init__(U, V, W, origin)