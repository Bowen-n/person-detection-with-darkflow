
import sys
import fcntl
import cv2
from darkflow.net.build import TFNet
from time import time as timer
from related.chassis import chassis

# load the trained model which is stord as .pb and .meta
options= {"pbLoad": "built_graph/tiny-yolo-voc-person.pb", "metaLoad": "built_graph/tiny-yolo-voc-person.meta",
          "threshold": 0.45, "demo": "camera", "gpu": 0.5}
tfnet = TFNet(options)

camera = cv2.VideoCapture(1) # get external camera
ch = chassis()
ch.open()

_, frame = camera.read() # get picture(array) from camera
height, width, _ = frame.shape
print(height)
print(width)

# buffers
in_buffer = []
pre_buffer = []
count = 0
start = timer()
tfnet.say('Press [ESC] to quit demo')

# loop by frame
while camera.isOpened():
	count += 1
	_, frame = camera.read()
	if frame is None:
		print('\nEnd of Video')
		break

	preprocessed = tfnet.framework.preprocess(frame) # preprocessing the array
	in_buffer.append(frame)
	pre_buffer.append(preprocessed)

	if count % 2 == 0:
		feed_dict = {tfnet.inp: pre_buffer}
		net_out = tfnet.sess.run(tfnet.out, feed_dict)

		for img, single_out in zip(in_buffer, net_out):
			postprocessed = tfnet.framework.postprocess(single_out, img, False)
			result = tfnet.return_predict(img) # a list of box(dictionary)

			if result:

				# get the biggest person in the picture
				person_max = {'x_1': 0, 'y_1': 0, 'x_2': 0, 'y_2': 0}
				
				for person_dict in result:
					if (person_dict['bottomright']['x'] - person_dict['topleft']['x']) > (person_max['x_2'] - person_max['x_1']) and \
							(person_dict['bottomright']['y'] - person_dict['topleft']['y']) > (person_max['y_2'] - person_max['y_1']):
						x_1, y_1 = person_dict['topleft']['x'], person_dict['topleft']['y']
						x_2, y_2 = person_dict['bottomright']['x'], person_dict['bottomright']['y']
						person_max = {'x_1': x_1, 'y_1': y_1, 'x_2': x_2, 'y_2': y_2}  

				print(person_max)
				# find the center of person
				x_center = (person_max['x_1'] + person_max['x_2']) / 2
				y_center = (person_max['y_1'] + person_max['y_2']) / 2

				# car controlling (follow person)
				if x_center > 0.3 * width and x_center < 0.7 * width:
					ch.moveStepForward(0.12)
				elif x_center < 0.3 * width:
					ch.moveStepRight(0.06)
				elif x_center > 0.7 * width:
					ch.moveStepLeft(0.06)

			# write out the image
			img_name = "web_stream/out/test.jpg"
			fp = open(img_name[:-3] + 'lock', 'w')
			fcntl.flock(fp, fcntl.LOCK_EX)
			cv2.imwrite(img_name, postprocessed)
			fcntl.flock(fp, fcntl.LOCK_UN)

		# Clear Buffers
		in_buffer = list()
		pre_buffer = list()

	if count % 5 == 0:
		sys.stdout.write('\r')
		sys.stdout.write('{0:3.3f} FPS'.format(count / (timer() - start)))
		sys.stdout.flush()

ch.close()
sys.stdout.write('\n')
camera.release()
