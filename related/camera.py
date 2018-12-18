import cv2
import fcntl


class VideoCamera(object):
    def __init__(self, cam_type):
        self.video = cv2.VideoCapture(cam_type)  # 0 or 1

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


class VideoFile(object):
    def __init__(self):
        # self.video = cv2.VideoCapture(0)
        pass

    def __del__(self):
        # self.video.release()
        pass

    def get_frame(self):
        fp = open('/tmp/webstream.lock', 'w')
        fcntl.flock(fp, fcntl.LOCK_EX)
        image = cv2.imread("/tmp/webstream.jpg")
        fcntl.flock(fp, fcntl.LOCK_UN)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    @staticmethod
    def feed(dataMat):
        fp = open('/tmp/webstream.lock', 'w')
        fcntl.flock(fp, fcntl.LOCK_EX)
        cv2.imwrite("/tmp/webstream.jpg", dataMat)
        fcntl.flock(fp, fcntl.LOCK_UN)
