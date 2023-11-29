import Wireframe_EKF as wf
import pygame
from operator import itemgetter
import readSensor_EKF as rs
from icm20948 import ICM20948
import numpy as np
import pandas as pd
import time
import glob
from colorama import Fore , Back


class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen """
    def __init__(self, width, height, wireframe):
        self.width = width
        self.height = height
        self.wireframe = wireframe
        # self.image_addresses = sorted(glob.glob(images_directory))
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Attitude Determination using Quaternions')
        self.background = (10,10,50)
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.frequency = 300
        

    def run(self, sensorInstance):
        """ Create a pygame screen until it is closed. """
        running = True
        loopRate = 100
        data = np.zeros(9)
        idx = 0
        start_time = 0

        while running:

            start_time = time.perf_counter()
            idx += 1

            magnetometer_x, magnetometer_y, magnetometer_z = sensorInstance.read_magnetometer_data()
            # magnetometer_x = magnetometer_x * 0.080
            # magnetometer_y = magnetometer_y * 0.080
            # magnetometer_z = magnetometer_z * 0.080

            accelerometer_x, accelerometer_y, accelerometer_z, gyro_x, gyro_y, gyro_z = sensorInstance.read_accelerometer_gyro_data()
            
            gyro_x = (gyro_x * 0.00875) / 180.0 * np.pi
            gyro_y = (gyro_y * 0.00875) / 180.0 * np.pi
            gyro_z = (gyro_z * 0.00875) / 180.0 * np.pi
            accelerometer_x = (accelerometer_x * 0.061) * 9.81
            accelerometer_y = (accelerometer_y * 0.061) * 9.81
            accelerometer_z = (accelerometer_z * 0.061) * 9.81
            # magnetometer_x = (magnetometer_x * 0.2) 
            # magnetometer_y = (magnetometer_y * 0.2) 
            # magnetometer_z = (magnetometer_z * 0.2) 
            # magnetometer_x = 0 
            # magnetometer_y = 0 
            # magnetometer_z = 0
            
            # data = [Gyroscope[idx,0],Gyroscope[idx,1],Gyroscope[idx,2],Accelerometer[idx,0],Accelerometer[idx,1],Accelerometer[idx,2],Magnetometer[idx,0],Magnetometer[idx,1],Magnetometer[idx,2]]
                      
            self.clock.tick(loopRate)
            # data = sensorInstance.getSerialData()  
            # self.wireframe.quatRotate([data[0], data[1], data[2]],
            #                             [data[3], data[4], data[5]],
            #                             [data[6], data[7], data[8]],
            #                             1/loopRate)
            self.wireframe.quatRotate([gyro_x,gyro_y,gyro_z],
                                    [ accelerometer_x, accelerometer_y, accelerometer_z],
                                    [magnetometer_x,-magnetometer_y, -magnetometer_z],
                                    1/loopRate)

            while (time.perf_counter()- start_time) < (1/self.frequency):
                pass
            
            if idx == self.frequency:
                # print(idx)
                print( Fore.GREEN + '\033[1m'"\n sample rate ------> "'\033[0m', 1/(time.perf_counter()-start_time))
                idx = 0

            self.display()
            pygame.display.flip()


    def display(self):
        """ Draw the wireframes on the screen. """
        self.screen.fill(self.background)
    
        # Get the current attitude
        yaw, pitch, roll = self.wireframe.getAttitude()
        self.messageDisplay("Yaw: %.1f" % yaw,
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0,
                            (220, 20, 60))      # Crimson
        self.messageDisplay("Pitch: %.1f" % pitch,
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0.05,
                            (0, 255, 255))     # Cyan
        self.messageDisplay("Roll: %.1f" % roll,
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0.1,
                            (65, 105, 225))    # Royal Blue
        self.messageDisplay("Sample Rate: 35",
                            self.screen.get_width()*0.75,
                            self.screen.get_height()*0.15,
                            (255, 255, 255))    # White
        print ( Fore.YELLOW + "\n yaw: %.1f  pitch: %.1f  roll: %.1f" %(yaw,pitch,roll) )
        # Transform nodes to perspective view
        dist = 5
        pvNodes = []
        pvDepth = []
        for node in self.wireframe.nodes:
            point = [node.x, node.y, node.z]
            newCoord = self.wireframe.rotatePoint(point)
            # print(newCoord)
            comFrameCoord = self.wireframe.convertToComputerFrame(newCoord)
            pvNodes.append(self.projectOthorgraphic(comFrameCoord[0], comFrameCoord[1], comFrameCoord[2],
                                                    self.screen.get_width(), self.screen.get_height(),
                                                    70, pvDepth))
            """
            pvNodes.append(self.projectOnePointPerspective(comFrameCoord[0], comFrameCoord[1], comFrameCoord[2],
                                                           self.screen.get_width(), self.screen.get_height(),
                                                           5, 10, 30, pvDepth))
            """

        # Calculate the average Z values of each face.
        avg_z = []
        for face in self.wireframe.faces:
            n = pvDepth
            z = (n[face.nodeIndexes[0]] + n[face.nodeIndexes[1]] +
                 n[face.nodeIndexes[2]] + n[face.nodeIndexes[3]]) / 4.0
            avg_z.append(z)
        # Draw the faces using the Painter's algorithm:
        for idx, val in sorted(enumerate(avg_z), key=itemgetter(1)):
            face = self.wireframe.faces[idx]
            pointList = [pvNodes[face.nodeIndexes[0]],
                         pvNodes[face.nodeIndexes[1]],
                         pvNodes[face.nodeIndexes[2]],
                         pvNodes[face.nodeIndexes[3]]]
            pygame.draw.polygon(self.screen, face.color, pointList)
        
    # One vanishing point perspective view algorithm
    def projectOnePointPerspective(self, x, y, z, win_width, win_height, P, S, scaling_constant, pvDepth):
        # In Pygame, the y axis is downward pointing.
        # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
        # This will result in y' = -y and z' = -z
        xPrime = x
        yPrime = -y
        zPrime = -z
        xProjected = xPrime * (S/(zPrime+P)) * scaling_constant + win_width / 2
        yProjected = yPrime * (S/(zPrime+P)) * scaling_constant + win_height / 2
        pvDepth.append(1/(zPrime+P))
        return (round(xProjected), round(yProjected))

    # Normal Projection
    def projectOthorgraphic(self, x, y, z, win_width, win_height, scaling_constant, pvDepth):
        # In Pygame, the y axis is downward pointing.
        # In order to make y point upwards, a rotation around x axis by 180 degrees is needed.
        # This will result in y' = -y and z' = -z
        xPrime = x
        yPrime = -y
        xProjected = xPrime * scaling_constant + win_width / 2
        yProjected = yPrime * scaling_constant + win_height / 2
        # Note that there is no negative sign here because our rotation to computer frame
        # assumes that the computer frame is x-right, y-up, z-out
        # so this z-coordinate below is already in the outward direction
        pvDepth.append(z)
        return (round(xProjected), round(yProjected))

    def messageDisplay(self, text, x, y, color):
        textSurface = self.font.render(text, True, color, self.background)
        textRect = textSurface.get_rect()
        textRect.topleft = (x, y)
        self.screen.blit(textSurface, textRect)

def initializeCube():
    block = wf.Wireframe()

    block_nodes = [(x, y, z) for x in (-0.8, 0.8) for y in (-1.5, 1.5) for z in (-0.1, 0.1)]
    node_colors = [(255, 255, 255)] * len(block_nodes)
    block.addNodes(block_nodes, node_colors)
    block.outputNodes()

    faces = [(0, 2, 6, 4), (0, 1, 3, 2), (1, 3, 7, 5), (4, 5, 7, 6), (2, 3, 7, 6), (0, 1, 5, 4)]
    colors = [(239, 71, 111), (247, 140, 107), (255, 209, 102), (6, 214, 160), (17, 138, 178), (7, 59, 76)]
    block.addFaces(faces, colors)
    block.outputFaces()

    return block


if __name__ == '__main__':
    # df = pd.read_csv("/home/jetson/imu/sensor_data_IMU_1_2023-10-28_09-02-35_Modified (copy).csv")
    # column_names_IMU = ["Timestamp","Accelerometer_X","Accelerometer_Y","Accelerometer_Z","Gyro_X","Gyro_Y","Gyro_Z","Magnetometer_X","Magnetometer_Y","Magnetometer_Z"]
    # IMU_data = np.array(df.loc[0:10000,column_names_IMU])
    # time = IMU_data[0:len(IMU_data),0]
    # Accelerometer = IMU_data[0:len(IMU_data),1:4]
    # Gyroscope = IMU_data[0:len(IMU_data),4:7]
    # Magnetometer = IMU_data[0:len(IMU_data),7:10]


    # print(seconds)

    icm_sensor = ICM20948(Model = "mpu9250") #Model = "mpu9250"
    block = initializeCube()
    pv = ProjectionViewer(800, 600, block)
    pv.run(icm_sensor)