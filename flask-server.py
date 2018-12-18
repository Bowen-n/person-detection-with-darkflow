from flask import Flask, render_template, Response
import cv2
import fcntl

app = Flask(__name__)


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


def gen():
    while True:
        # 打开文件锁
        fp = open('web_stream_picture/out/test.lock', 'w')
        fcntl.flock(fp, fcntl.LOCK_EX)
        # 读取测试过的图片
        processed_image = cv2.imread('web_stream_picture/out/test.jpg')
        fcntl.flock(fp, fcntl.LOCK_UN)
        ret, jpeg = cv2.imencode('.jpg', processed_image)
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(host='192.168.199.116', debug=True, port=5000)