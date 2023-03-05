import argparse
from asyncio import run
from collections import Counter
from distutils import core
from platform import processor
from pyexpat import model
import cv2
import sys
import vision

s=0
if sys.argv(s)>1:
    s=sys.argv[1]



cap = cv2.VideoCapture(s)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 50)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 50)

row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

base_options = core.BaseOptions(
    file_name=model, use_coral='enable_edgetpu', num_threads='num_threads')
detection_options = processor.DetectionOptions(
    max_results=3, score_threshold=0.3)
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

while cap.isOpened():
  success, image = cap.read()
  if not success:
    sys.exit(
        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    )

  Counter += 1
  image = cv2.flip(image, 1)


  detection_result = detector.detect(input_tensor)
  


def main():
  parser = argparse.ArgumentParser(
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--model',
                      help='Path of the object detection model.',
                      required=False,
                      default='pothole_label.tflite')
  parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument('--frameWidth',
                          help='Width of frame to capture from camera.',
                          required=False,
                          type=int,
                          default=640)
  parser.add_argument('--frameHeight',
                      help='Height of frame to capture from camera.',
                      required=False,
                      type=int,
                      default=480)
  parser.add_argument('--numThreads',
                      help='Number of CPU threads to run the model.',
                      required=False,
                      type=int,
                      default=4)
  parser.add_argument('--enableEdgeTPU',
                      help='Whether to run the model on EdgeTPU.',
                      action='store_true',
                      required=False,
                      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))
  

if detection_result.shape[0]:
    cv2.imwrite(f'{counter}.jpg', image)
if __name__ == '__main__':
  main()
  
   