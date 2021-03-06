import numpy as np
import cv2
import skimage
import math
import pyaudio
import threading
from skimage import measure
from skimage.morphology import rectangle

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target, freq, stream, *args, **kwargs):
        super(StoppableThread, self).__init__()
        self.target = target
        self._stop = threading.Event()
        self.freq = freq
        self.stream = stream

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def update_freq(self, freq):
        self.freq = freq

    def run(self):
        while not self.stopped():
            if self.target:
                self.target(self, self.stream,self.freq)
        #    else:
        #        raise Exception('No target function given')
            #self.stop().wait(1)
            #self.stop_event.wait(self.sleep_time)

def startContinuousTone(self, stream, freq):
    play_tone(stream, freq, 0.075)
    #if self.stopped():
    #    cont = False

def sine(frequency, length, rate):
    length = int(length * rate)
    # add a volume in here, fast ramp up (to get rid of clicks)
    rampUp = np.linspace(0.0,1.0,int(math.floor(length/2.0)))
    rampDown = np.linspace(1.0,0.0,length - int(math.floor(length/2.0)))
    #volControl = np.concatenate(rampUp,rampDown)
    volControl = np.hstack((rampUp,rampDown))
    factor = float(frequency) * (math.pi * 2) / rate
    return np.multiply(np.sin(np.arange(length) * factor), volControl)

def play_tone(stream, frequency=440, length=1.0, rate=44100):
    chunks = []
    chunks.append(sine(frequency, length, rate))

    chunk = np.concatenate(chunks) * 0.25

    stream.write(chunk.astype(np.float32).tostring())

def genSound(freq):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

    play_tone(stream, freq, 0.3)

    stream.close()
    p.terminate()
    return

def locToFreq(x, imWidth):
    freqLBound = 261.63
    freqUBound = 587.33
    xNorm = x/imWidth
    freq = freqLBound + xNorm * (freqUBound - freqLBound)
    return freq

cap = cv2.VideoCapture(0)

p1 = pyaudio.PyAudio()
stream1 = p1.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)

p2 = pyaudio.PyAudio()
stream2 = p2.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)

p3 = pyaudio.PyAudio()
stream3 = p3.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)

bg = None
bgSet = False
hist = 80
cnt = 0.0
t1 = None
t2 = None
t3 = None
pts = []
x = -1
y = -1
scale = 0.6

while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()

    frame1 = cv2.flip(frame1,1)
    frame = cv2.resize(frame1, (0,0), fx=scale, fy=scale)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        bgSet = True
        bgImg = np.zeros_like(frame).astype(np.float16)

    frameDisplay = frame1
    idx = 0
    for pt in pts:
        if idx == 0:
            cv2.circle(frameDisplay,(int(pt[0] * 1/scale),int(pt[1] * 1/scale)),10,(0,0,255),-1)
        elif idx == 1:
            cv2.circle(frameDisplay,(int(pt[0] * 1/scale),int(pt[1] * 1/scale)),10,(0,255,0),-1)
        else:
            cv2.circle(frameDisplay,(int(pt[0] * 1/scale),int(pt[1] * 1/scale)),10,(255,0,0),-1)
        idx = idx + 1;


    # Display the resulting frame
    cv2.imshow('frame',frameDisplay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if bgSet:
        if cnt < hist:
            cnt = cnt + 1.0
            bgImg = bgImg + frame
            bg = bgImg/cnt
            cv2.imshow('background', bg.astype(np.uint8))
        # print type(gray-bg)
        #print numpy.size(gray,0)
        #print numpy.size(gray,1)

        # bgThresh = np.zeros((np.size(gray,0),np.size(gray,1)))
        diff = np.absolute(frame - bg)

        cv2.imshow('colorDiff',diff.astype(np.uint8))
        # print np.max(gray)
        # print np.min(gray)


        high_values_indicesR = diff[:,:,0] > 40.0  # Where values are high
        high_values_indicesG = diff[:,:,1] > 40.0  # Where values are high
        high_values_indicesB = diff[:,:,2] > 40.0  # Where values are high
        # array_np[low_values_indices] = 0  # All low values set to 0
        bgThresh = np.zeros((np.size(frame[:,:,0],0),np.size(frame[:,:,0], 1)))
        bgThresh[high_values_indicesR] = 1
        bgThresh[high_values_indicesG] = 1
        bgThresh[high_values_indicesB] = 1
        # bgThresh = skimage.morphology.binary_closing(bgThresh,rectangle(20,20))
        bgThresh = skimage.morphology.binary_opening(bgThresh,rectangle(10, 10))
        lblImg = skimage.measure.label(bgThresh, connectivity=2)

        #bgCC = skimage.morphology.remove_small_objects(lblImg,250,1,False)
        bgCC = lblImg
        #lblImg = skimage.measure.label(bgCC, connectivity=2)

        nonZeroIdx = bgCC > 0  # Where values are low
        mask = np.zeros((np.size(frame[:,:,0],0),np.size(frame[:,:,0],1)))
        mask[nonZeroIdx] = 1
        # cv2.imshow('diff',mask)

        regions = skimage.measure.regionprops(lblImg)

        area1 = 0
        area2 = 0
        area3 = 0
        savedReg1 = None
        savedReg2 = None
        savedReg3 = None

        foundReg1 = False
        foundReg2 = False
        foundReg3 = False

        if np.max(bgCC) > 0:
            bgCC = (bgCC * float((255.0/float(np.max(bgCC))))).astype(np.uint8)
            dispC = cv2.applyColorMap(bgCC,cv2.COLORMAP_JET)
            cv2.imshow("diff",dispC)
        else:
            cv2.imshow('diff', mask)

        pts = []
        for props in regions:
            if props.area < 250:
                continue
            if props.area > area1:
                if foundReg2:
                    savedReg3 = savedReg2
                    foundReg3 = True

                if foundReg1:
                    savedReg2 = savedReg1
                    foundReg2 = True

                area1 = props.area
                savedReg1 = props
                foundReg1 = True
                #print 'found Largest Region'

            elif props.area > area2:
                if foundReg2:
                    savedReg3 = savedReg2
                    foundReg3 = True
                    area3 = area2

                area2 = props.area
                savedReg2 = props
                foundReg2 = True

            elif props.area > area3:
                area3 = props.area
                savedReg3 = props
                foundReg3 = True


        if foundReg1:
            y,x = savedReg1.centroid
            pts.append((x,y))

            if t1 is not None:
                t1.update_freq(locToFreq(x, np.size(frame[:, :, 0], 0)))
            else:
                t1 = StoppableThread(target=startContinuousTone, freq=locToFreq(x, np.size(frame[:, :, 0], 0)), stream=stream1)
                t1.start()
        else:
            if t1 is not None:
                t1.stop()
                t1 = None

        if foundReg2:
            y,x = savedReg2.centroid
            pts.append((x,y))

            if t2 is not None:
                t2.update_freq(locToFreq(x, np.size(frame[:, :, 0], 0)))
            else:
                t2 = StoppableThread(target=startContinuousTone, freq=locToFreq(x, np.size(frame[:, :, 0], 0)), stream=stream2)
                t2.start()
        else:
            if t2 is not None:
                t2.stop()
                t2 = None

        if foundReg3:
            y,x = savedReg3.centroid
            pts.append((x,y))

            if t3 is not None:
                t3.update_freq(locToFreq(x, np.size(frame[:, :, 0], 0)))
            else:
                t3 = StoppableThread(target=startContinuousTone, freq=locToFreq(x, np.size(frame[:, :, 0], 0)), stream=stream3)
                t3.start()
        else:
            if t3 is not None:
                t3.stop()
                t3 = None







# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

stream1.close()
p1.terminate()

stream2.close()
p2.terminate()

stream3.close()
p3.terminate()


