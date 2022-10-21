import cv2
from djitellopy import tello
import cvzone
import multiprocessing
from time import sleep


me = tello.Tello()
me.connect()


class DroneTakeoff :
    def __init__(self, model_config, model_weight, detectionThreshold, NMSThreshold, droneVelocity):
        super(DroneTakeoff, self).__init__()
        self.model_config = model_config
        self.model_weight = model_weight
        self.detectionThreshold = detectionThreshold
        self.NMSThreshold = NMSThreshold
        self.droneVelocity = droneVelocity

    def droneTakeOffAndDetect(self):
        thres = self.detectionThreshold
        nmsThres = self.NMSThreshold
        classFile = 'coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().split('\n')
        print(classNames)

        configPath = self.model_config
        weightsPath = self.model_weight

        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        print(me.get_battery())
        me.streamoff()
        me.streamon()

        me.takeoff()

        while True:
            print("Entered the while...")
            img = me.get_frame_read().frame
            classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
            try:
                for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cvzone.cornerRect(img, box)
                    cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                                (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 255, 0), 2)
            except:
                pass

            me.send_rc_control(0, 0, 0, 0)

            cv2.imshow("Image", img)
            sleep(2)

            print("Drone Moving with detection....")

            vel = self.droneVelocity
            sleep(3)
            me.send_rc_control(0, vel, 0, 0)
            sleep(3)
            me.send_rc_control(0, 0, 0, 60)
            sleep(2)
            me.send_rc_control(0, vel, 0, 0)
            sleep(3)
            me.send_rc_control(0, 0, 0, 60)
            sleep(2)
            me.send_rc_control(0, vel, 0, 0)
            sleep(3)
            me.send_rc_control(0, 0, 0, 60)
            sleep(2)
            me.send_rc_control(0, vel, 0, 0)
            sleep(3)
            me.send_rc_control(0, 0, 0, 60)
            sleep(2)
            me.send_rc_control(0, 0, 0, 0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                me.land()


if __name__ == '__main__':
    WEIGHTS = "frozen_inference_graph.pb"
    CONFIGURATION = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    DETECTION_THRESHOLD = 0.45
    NMS_THRESHOLD = 0.25
    DRONE_VELOCITY = 50

    TelloDrone = DroneTakeoff(CONFIGURATION, WEIGHTS, DETECTION_THRESHOLD, NMS_THRESHOLD, DRONE_VELOCITY)
    TelloDrone.droneTakeOffAndDetect()









