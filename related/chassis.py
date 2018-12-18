#!/usr/bin/python3

import serial
import time

class chassis(object):
    __DIRECTION_FORWARD = 1
    __DIRECTION_BACKWARD = 2
    __DIRECTION_RIGHT = 3
    __DIRECTION_LEFT = 4
    __DIRECTION_STOP = 5

    def __init__(self):
        self.ser = serial.Serial("/dev/ttyTHS2",115200,timeout=0.5)
        # self.ser = None
        self.timestep = 0.085
        self.minDuration = 0.01

    def open(self):
        # self.ser.open()
        pass

    def __stop(self):
        cmd="+IPD,0,1:5"
        self.ser.write(cmd.encode())
        print(cmd)

    def __move(self, direction, duration):
        if duration <self.minDuration:
            return
        while duration > 0:
            cmd="+IPD,0,1:%d"%direction
            self.ser.write(cmd.encode())
            print(cmd)    
            if duration > self.timestep:
                time.sleep(self.timestep)
            else:
                time.sleep(duration)
                self.__stop()
            duration -= self.timestep
        
    def moveStepForward(self, duration):
        self.__move(chassis.__DIRECTION_FORWARD, duration)
    def moveStepBackward(self, duration):
        self.__move(chassis.__DIRECTION_BACKWARD, duration)
    def moveStepRight(self, duration):
        self.__move(chassis.__DIRECTION_RIGHT, duration)
    def moveStepLeft(self, duration):
        self.__move(chassis.__DIRECTION_LEFT, duration)
    def moveStop(self):
        self.__stop()

    def close(self):
        self.ser.close()