import Kalman_EKF as km
import numpy as np
from scipy.spatial.transform import Rotation as R

# Node stores each point of the block
class Node:
    def __init__(self, coordinates, color):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.color = color

# Face stores 4 nodes that make up a face of the block
class Face:
    def __init__(self, nodes, color):
        self.nodeIndexes = nodes
        self.color = color

# Wireframe stores the details of a block
class Wireframe:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.faces = []
        self.sys = km.System()

    def addNodes(self, nodeList, colorList):
        for node, color in zip(nodeList, colorList):
            self.nodes.append(Node(node, color))

    def addFaces(self, faceList, colorList):
        for indexes, color in zip(faceList, colorList):
            self.faces.append(Face(indexes, color))

    def quatRotate(self, w, a, m, dt):
        self.sys.predict(w, dt)
        self.sys.update(a, m)

    def rotatePoint(self, point):
        rotationMat = km.getRotMat(self.sys.xHat[0:4])
        # print('rotation matrix ----', rotationMat)
        # r = R.from_matrix(rotationMat)
        # euler1 = r.as_euler('xyz', degrees=True)
        # euler2 = r.as_euler('yxz', degrees=True)
        # euler1_from = R.from_euler('yxz',euler1, degrees=True)
        # euler2_from = R.from_euler('yxz',euler2, degrees=True)
        # euler1_rot = euler1_from.as_matrix()
        # euler2_rot = euler2_from.as_matrix()
        # print('xyz',rotationMat)
        # print('yxz',euler2_rot)
        # print("Euler1 Angles" , euler1)
        # print("Euler2 Angles" , euler2)
        rot_angle = np.pi / 2
        roll_rotation = np.array ([[1,0,0],[0,np.cos(rot_angle),-np.sin(rot_angle)],[0,np.sin(rot_angle),np.cos(rot_angle)]])
        mirror_rotation = np.array ([[1,0,0],[0,1,0],[0,0,1]])
        # print('roll rotation ------', roll_rotation)
        # print(roll_rotation.shape)
        # print(rotationMat.shape)
        rotated_point = np.matmul(rotationMat, point)
        # print('rotated point', rotated_point)
        # print('final point', np.matmul(roll_rotation, rotated_point))
        # rotated_point2 = np.matmul(roll_rotation, rotated_point)
        # rotated_point3 = np.matmul(mirror_rotation, rotated_point)
        return rotated_point

    def convertToComputerFrame(self, point):
        computerFrameChangeMatrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        return np.matmul(computerFrameChangeMatrix, point)

    def getAttitude(self):
        Eulerangles = km.getEulerAngles(self.sys.xHat[0:4])
        Eulerangles2 = [0,0,0]
        # print(Eulerangles2)
        return Eulerangles

    def outputNodes(self):
        print("\n --- Nodes --- ")
        for i, node in enumerate(self.nodes):
            print(" %d: (%.2f, %.2f, %.2f) \t Color: (%d, %d, %d)" %
                 (i, node.x, node.y, node.z, node.color[0], node.color[1], node.color[2]))

    def outputFaces(self):
        print("\n --- Faces --- ")
        for i, face in enumerate(self.faces):
            print("Face %d:" % i)
            print("Color: (%d, %d, %d)" % (face.color[0], face.color[1], face.color[2]))
            for nodeIndex in face.nodeIndexes:
                print("\tNode %d" % nodeIndex)