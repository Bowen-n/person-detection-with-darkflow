
import sys
from darkflow.net.build import TFNet
import cv2
from time import time as timer
import fcntl
from related.chassis import chassis

# 神经网络参数选项
options= {"pbLoad": "built_graph/yolo-tiny-new.pb", "metaLoad": "built_graph/yolo-tiny-new.meta",
          "threshold": 0.35, "demo": "camera", "gpu": 0.5}
tfnet = TFNet(options)
type = 0 # 标注摄像头种类0或1
queue = 2 # 每次处理的照片队列长度
camera = cv2.VideoCapture(type) # 获取摄像头实例
ch = chassis()
ch.open()


# 确认摄像头种类
if type == 0:  # camera window
	cv2.namedWindow('', 0)
	_, frame = camera.read() # frame是np.ndarray
	# print(frame)
	height, width, _ = frame.shape
	cv2.resizeWindow('', width, height)
	print(height)
	print(width)
else:
	_, frame = camera.read()
	height, width, _ = frame.shape
	print(height)
	print(width)




# buffers for demo in batch
buffer_inp = list()
buffer_pre = list()
elapsed = int()
start = timer()
tfnet.say('Press [ESC] to quit demo')


# Loop through frames
while camera.isOpened():
	elapsed += 1
	_, frame = camera.read()
	if frame is None:
		print('\nEnd of Video')
		break
	preprocessed = tfnet.framework.preprocess(frame) # 将多维数组进行预处理
	buffer_inp.append(frame)
	buffer_pre.append(preprocessed)

	# Only process and imshow when queue is full
	if elapsed % queue == 0:
		feed_dict = {tfnet.inp: buffer_pre}
		net_out = tfnet.sess.run(tfnet.out, feed_dict)
		for img, single_out in zip(buffer_inp, net_out):
			postprocessed = tfnet.framework.postprocess(single_out, img, False)


			result = tfnet.return_predict(img) # result为boxinfo的列表，列表中的元素为box字典
			# print(result)
			if result:
				person_max = {'x_1': 0, 'y_1': 0, 'x_2': 0, 'y_2': 0}
				for person_dict in result:
					if (person_dict['bottomright']['x'] - person_dict['topleft']['x']) > (person_max['x_2'] - person_max['x_1']) and \
							(person_dict['bottomright']['y'] - person_dict['topleft']['y']) > (person_max['y_2'] - person_max['y_1']):
						x_1, y_1 = person_dict['topleft']['x'], person_dict['topleft']['y']
						x_2, y_2 = person_dict['bottomright']['x'], person_dict['bottomright']['y']
						person_max = {'x_1': x_1, 'y_1': y_1, 'x_2': x_2, 'y_2': y_2}  # 加载每张图片中人的大小尺寸
				print(person_max)
				x_center, y_center = (person_max['x_1'] + person_max['x_2']) / 2, (person_max['y_1'] + person_max['y_2']) / 2
				# 控制小车移动
				if x_center > 0.3 * width and x_center < 0.7 * width:
					ch.moveStepForward(0.2)
				elif x_center < 0.3 * width:
					ch.moveStepLeft(0.2)
				elif x_center > 0.7 * width:
					ch.moveStepRight(0.2)

			if type == 0:  # camera window
				cv2.imshow('', postprocessed)
			elif type == 1:
				# 画出处理后的图像
				img_name = "web_stream/out/test.jpg"
				fp = open(img_name[:-3] + 'lock', 'w')
				fcntl.flock(fp, fcntl.LOCK_EX)
				cv2.imwrite(img_name, postprocessed)
				fcntl.flock(fp, fcntl.LOCK_UN)


		# Clear Buffers
		buffer_inp = list()
		buffer_pre = list()
	if elapsed % 5 == 0:
		sys.stdout.write('\r')
		sys.stdout.write('{0:3.3f} FPS'.format(
			elapsed / (timer() - start)))
		sys.stdout.flush()
	if type == 0:  # camera window
		choice = cv2.waitKey(1)
		if choice == 27: break

ch.close()
sys.stdout.write('\n')
camera.release()
if type == 0:  # camera window
	cv2.destroyAllWindows()
