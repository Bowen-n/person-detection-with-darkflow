from flask import Flask, render_template, Response
from flask import request
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.pardir))
print(os.path.abspath(os.path.pardir))
print(os.path.dirname(os.path.abspath(__file__)))

from camera import VideoCamera, VideoFile, VideoObjectDetection


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    print("0000")
    return Response(gen(VideoObjectDetection()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
