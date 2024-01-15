import cv2
import RPi.GPIO as GPIO

# gnd(l4),pwm1(l5),dir2(l10),dir1(l13/r8),pwm2(r6)
PWM_Pin_1 = 12
PWM_Pin_2 = 18
Dir_Pin_1 = 23
Dir_Pin_2 = 25

i = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(PWM_Pin_1, GPIO.OUT)
GPIO.setup(PWM_Pin_2, GPIO.OUT)
GPIO.setup(Dir_Pin_1, GPIO.OUT)
GPIO.setup(Dir_Pin_2, GPIO.OUT)

p1 = GPIO.PWM(PWM_Pin_1, 490)
p2 = GPIO.PWM(PWM_Pin_2, 490)

p1.start(0)
p2.start(0)

# window size
H = 400
W = 800

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

def turnright(speed):
    GPIO.output(Dir_Pin_1, GPIO.HIGH)
    GPIO.output(Dir_Pin_2, GPIO.LOW)
    p1.ChangeDutyCycle(speed)
    p2.ChangeDutyCycle(speed)
    
def turnleft(speed):
    GPIO.output(Dir_Pin_1, GPIO.LOW)
    GPIO.output(Dir_Pin_2, GPIO.HIGH)
    p1.ChangeDutyCycle(speed)
    p2.ChangeDutyCycle(speed)
    
def accel():
    GPIO.output(Dir_Pin_1, GPIO.HIGH)
    GPIO.output(Dir_Pin_2, GPIO.HIGH)
    p1.ChangeDutyCycle(20)
    p2.ChangeDutyCycle(20)
    
def goBack():
    GPIO.output(Dir_Pin_1, GPIO.LOW)
    GPIO.output(Dir_Pin_1, GPIO.LOW)
    p1.ChangeDutyCycle(30)
    p2.ChangeDutyCycle(30)
    
def stop():
    GPIO.output(Dir_Pin_1, GPIO.LOW)
    GPIO.output(Dir_Pin_2, GPIO.LOW)
    p1.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(0)

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,W)
    cap.set(4,H)
    #cap.set(10,70)


    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.50, 0.2, objects=['person'])
        print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
        
        #fillter obj for biggest box
        '''for obj in objectInfo:
            box, className = obj
            x, y, width, height = box
            size = (width*height/H/W*100)
            if size < preObjSize:
                obj = 0
            else:
                preObjS'''
        
        for obj in objectInfo:
            box, className = obj
            x, y, width, height = box # xy of upper left corner of box 0,0 at top left corner
            # print(f"Detected {className} at position ({x+width/2}, {y+height/2}), {width*height/H/W*100}percent")
            square_mid_x = x+width/2
            square_mid_y = y+height/2
            
            print(square_mid_x)
            
            if square_mid_x < 1.5*W/5:
                print("person in LEFT")
                speed = 0.1*(W/2 - square_mid_x)
                #if speed < 20:
                #    speed = 0
                if speed > 50:
                    speed = 50;
                print(speed)
                turnleft(speed)
            elif square_mid_x > 3.5*W/5:
                print("person in RIGHT")
                speed = 0.1*(square_mid_x-W/2)
                #if speed < 20:
                #    speed = 0
                if speed > 50:
                    speed = 50;
                print(speed)
                turnright(speed)
            else:
                print("person in FRONT")
                percent = (height*width/H/W*100)
                print(percent)
                stop()
                if (percent < 35):
                    print("goForward")
                    accel()
                #elif percent > 55:
                #    print("goBack")
                #    goBack()