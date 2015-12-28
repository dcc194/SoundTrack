import numpy as np
import cv2
import skimage
import math
import pyaudio
import threading
from skimage import morphology
from skimage import measure
from skimage import filters
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.morphology import rectangle
import matplotlib

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target=None,  **kwargs):
        super(StoppableThread, self, target=None,  **kwargs).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while not self.stopped():
            if self.target:
                self.target()
            else:
                raise Exception('No target function given')
            self.stop_event.wait(self.sleep_time)

def startContinuousTone(self,stream,freq):
    play_tone(stream,freq,0.1)
    if self.stopped():
        cont = False

def sine(frequency, length, rate):
    length = int(length * rate)
    # add a volume in here, fast ramp up (to get rid of clicks)

    factor = float(frequency) * (math.pi * 2) / rate
    return np.sin(np.arange(length) * factor)


def play_tone(stream, frequency=440, length=1, rate=44100):
    chunks = []
    chunks.append(sine(frequency, length, rate))

    chunk = np.concatenate(chunks) * 0.25

    stream.write(chunk.astype(np.float32).tostring())

def genSound(freq):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

    play_tone(stream,freq,0.3)

    stream.close()
    p.terminate()
    return






def locToFreq(x,imWidth):
    freqLBound = 261.63
    freqUBound = 587.33
    xNorm = x/imWidth
    freq = freqLBound + xNorm * (freqUBound - freqLBound)
    return freq

cap = cv2.VideoCapture(0)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,channels=1, rate=44100, output=1)

bg = None
bgSet = False
hist = 80
cnt = 0.0
threads = []
pts = []
x = -1
y = -1
scale = 0.6
while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()

    frame1 = cv2.flip(frame1,1)
    frame = cv2.resize(frame1, (0,0), fx=scale, fy=scale)
    # fgmask = fgbg.apply(frame1)
    # cv2.imshow('NewMask',fgmask)



    # rIm = frame[:,:,0]
    # gIm = frame[:,:,1]
    # bIm = frame[:,:,2]



    #frame = np.zeros_like(frame1)
    # Our operations on the frame come here
    #frame[:,:,0] = median(frame1[:,:,0],rectangle(10,10))
    #frame[:,:,1] = median(frame1[:,:,1],rectangle(10,10))
    #frame[:,:,2] = median(frame1[:,:,2],rectangle(10,10))

    # gray = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

    if cv2.waitKey(1) & 0xFF == ord('b'):
        # bg = gray
        bgSet = True
        bgImg = np.zeros_like(frame).astype(np.float16)
        #cv2.imshow('background', bg)

    frameDisplay = frame1
    #if x > -1:
    for pt in pts:
        cv2.circle(frameDisplay,(int(pt[0] * 1/scale),int(pt[1] * 1/scale)),10,(0,0,255),-1)

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

        # cv2.imshow('diff',diff.astype(int))
        # print np.max(gray)
        # print np.min(gray)


        high_values_indicesR = diff[:,:,0] > 20.0  # Where values are high
        high_values_indicesG = diff[:,:,1] > 20.0  # Where values are high
        high_values_indicesB = diff[:,:,2] > 20.0  # Where values are high
        # array_np[low_values_indices] = 0  # All low values set to 0
        bgThresh = np.zeros((np.size(frame[:,:,0],0),np.size(frame[:,:,0],1)))
        bgThresh[high_values_indicesR] = 1
        bgThresh[high_values_indicesG] = 1
        bgThresh[high_values_indicesB] = 1
        # bgThresh = skimage.morphology.binary_closing(bgThresh,rectangle(15,15))
        bgThresh = skimage.morphology.binary_opening(bgThresh,rectangle(30,30))
        lblImg = skimage.measure.label(bgThresh,connectivity=2)

        bgCC = skimage.morphology.remove_small_objects(lblImg,150,1,False)

        nonZeroIdx = bgCC > 0  # Where values are low
        mask = np.zeros((np.size(frame[:,:,0],0),np.size(frame[:,:,0],1)))
        mask[nonZeroIdx] = 1
        # cv2.imshow('diff',mask)

        regions = skimage.measure.regionprops(bgCC)

        largest = 0
        savedReg = None
        foundReg = False

        cv2.imshow('diff',mask)
        pts = []
        for props in regions:
            if props.area > largest:
                largest = props.area
                savedReg = props
                foundReg = True
                #print 'found Largest Region'


        if foundReg:
            y,x = savedReg.centroid
            pts.append((x,y))


            for th in threads:
                th.stop()

            #t = StoppableThread()
            t = StoppableThread(target=startContinuousTone, args=(stream,locToFreq(x,np.size(frame[:,:,0],0))))
            # t.run(stream,locToFreq(x,np.size(frame[:,:,0],0)))
            # t = StoppableThread()
            # t = threading.Thread(target=play_tone, args=(stream,locToFreq(x,np.size(frame[:,:,0],0)),0.3))
            threads.append(t)
            t.start()






        # if foundReg:
        #     y,x = savedReg.centroid
        #
        #     # cv2.circle(mask,(int(x),int(y)),10,(0,0,255),-1)
        #     #cv2.imshow('frame',frame)
        #
        #     cv2.imshow('diff',mask)
        #
        #     t = threading.Thread(target=genSound, args=(locToFreq(x,np.size(frame[:,:,0],0)),))
        #     # for thread in threads:
        #     #     thread.stop()
        #     threads.append(t)
        #
        #     t.start()







# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

stream.close()
p.terminate()


