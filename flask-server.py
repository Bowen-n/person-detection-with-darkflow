from flask import Flask, render_template, Response
import cv2
import fcntl

app = Flask(__name__)

@app.route('/') 
def index():
    return render_template('index.html')

def gen():
    while True:
        fp = open('web_stream/out/test.lock', 'w')
        fcntl.flock(fp, fcntl.LOCK_EX)
        boxed_image = cv2.imread('web_stream/out/test.jpg')
        fcntl.flock(fp, fcntl.LOCK_UN)
        ret, jpeg = cv2.imencode('.jpg', boxed_image)
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='172.18.16.92', debug=True, port=5000)
