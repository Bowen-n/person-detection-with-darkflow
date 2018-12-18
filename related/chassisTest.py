#!/usr/bin/python3

from chassis import chassis

ch = chassis()
ch.open()
for i in range(0, 10):
    ch.moveStepBackward(0.2)
ch.close()