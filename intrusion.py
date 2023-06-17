import cv2
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import tkinter as tk

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

# Set up GPIO for the buzzer
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Set up labels
INTRUSION_LABEL = "Intrusion Detected"
NO_INTRUSION_LABEL = "No Intrusion"

# Load object detection model and class names
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Set minimum distance threshold for triggering the buzzer
MIN_DISTANCE = 100  # Adjust this value according to your needs

# Define the animal classes to detect
animal_classes = ["elephant", "cow", "pig"]

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow camera to warm up
time.sleep(2)
intrusion_detected = False

# Create tkinter window
window = tk.Tk()
window.title("Intrusion Detection")
window.geometry("400x200")

# Create label widget
label = tk.Label(window, text=NO_INTRUSION_LABEL, font=("Arial", 24), pady=50)
label.pack()

# Function to update the label
def update_label():
    if intrusion_detected:
        label.config(text=INTRUSION_LABEL)
    else:
        label.config(text=NO_INTRUSION_LABEL)

# Main loop
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    result, objectInfo = getObjects(img, 0.45, 0.2, objects=animal_classes)

    intrusion_detected = False  # Reset the flag for each frame

    for _, (box, className) in enumerate(objectInfo):
        if className in animal_classes and box[2] > MIN_DISTANCE:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            intrusion_detected = True
            time.sleep(1)
            GPIO.output(BUZZER_PIN, GPIO.LOW)

    cv2.imshow("Output", img)
    rawCapture.truncate(0)

    if cv2.waitKey(1) == ord("q"):
        break

    update_label()  # Update the tkinter label
    window.update()  # Update the tkinter window

cv2.destroyAllWindows()
camera.close()
GPIO.cleanup()
window.mainloop()
