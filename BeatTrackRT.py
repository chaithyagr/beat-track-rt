import RPi.GPIO as GPIO
import pyaudio
import time
import numpy as np
import threading
import onset_timer
import playbeep
import os
import wave
import glob
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)

class BeatTrackRT(object):
    def __init__(self, device=None, rate=None, chunk=1024):
        """fire up the BeatTrackRT class."""
        self.p = pyaudio.PyAudio()
        self.chunk = chunk  # 2048 # number of data points to read at a time
        self.device = device
        self.adaptval = 15
        self.conver = chunk  # Length of step size...
        self.tdf = []
        self.fs = 48000
        self.rate = self.fs #rate
        self.time_stamps = []
        self.acorrwin = 3 * self.fs / chunk
        self.acorrmaxwin = 3 * self.fs / chunk
        self.bpm = []
        self.lentdf = 300 * self.fs / chunk
        self.thresh = 0.1
        self.C = []  # np.zeros((self.acorrwin,1))
        self.Start = []
        self.lastbeat = []
        self.firstrunflag = 1
        self.BT = []
        self.acorr = []
        self.TV = True
        self.samples = 2048
        self.rayparam = np.round(43 * (512 / 512))
        self.abpm = []
        self.bepm = []
        self.bepmdebug = []
        self.lbdebug = []
        self.ended=False

    def valid_low_rate(self, device):
        """set the rate to the lowest supported audio rate."""
        for testrate in [self.fs]:
            if self.valid_test(device, testrate):
                return testrate
        print("SOMETHING'S WRONG! I can't figure out how to use DEV", device)
        return None

    def valid_test(self, device, rate=48000):
        """given a device ID and a rate, return TRUE/False if it's valid."""
        try:
            self.info = self.p.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False
            stream = self.p.open(format=pyaudio.paInt16, channels=1,
                                 input_device_index=device, frames_per_buffer=self.chunk,
                                 rate=int(self.info["defaultSampleRate"]), input=True)
            stream.close()
            return True
        except:
            return False

    def valid_input_devices(self):
        mics = []
        for device in range(self.p.get_device_count()):
            if self.valid_test(device):
                mics.append(device)
        if len(mics) == 0:
            print("no microphone devices found!")
        else:
            print("found %d microphone devices: %s" % (len(mics), mics))
        return mics

    def initiate(self):
        """run this after changing settings (like rate) before recording"""
        if self.device is None:
            self.device = self.valid_input_devices()[0]  # pick the first one
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)
        if not self.valid_test(self.device, self.rate):
            print("guessing a valid microphone device/rate...")
            self.device = self.valid_input_devices()[0]  # pick the first one
            self.rate = self.valid_low_rate(self.device)
        self.datax = np.arange(self.chunk) / float(self.rate)
        msg = 'recording from "%s" ' % self.info["name"]
        msg += '(device %d) ' % self.device
        msg += 'at %d Hz' % self.rate
        print(msg)

    def close(self):
        """gently detach from things."""
        print(" -- sending stream termination command...")
        self.keepRecording = False  # the threads should self-close
        while (self.t.isAlive()):  # wait for all threads to close
            time.sleep(.1)
        self.stream.stop_stream()
        self.p.terminate()

    def stream_readchunk(self):
        """"reads some audio and re-launches itself"""
        try:
            self.data = np.fromstring(self.stream.read(self.chunk), dtype=np.int16) / 1024
        except Exception as E:
            print(" -- exception! terminating...")
            print(E, "\n" * 5)
            self.keepRecording = False
        if self.keepRecording:
            self.stream_thread_new()
        else:
            self.stream.close()
            self.p.terminate()
            print(" -- stream STOPPED")

    def stream_thread_new(self):
        self.t = threading.Thread(target=self.stream_readchunk)
        self.t.start()

    def stream_thread_onset(self):
        self.t2 = threading.Thread(target=self.rt_onset)
        self.t2.start()

    def stream_thread_start_beatseq(self):
        self.t4 = threading.Thread(target=self.printbeat)
        self.t4.start()
	
    def stream_gpio_led(self,x=100):
        self.t5 = threading.Thread(target=self.gpio_led)
        self.t5.start()
	
    def stream_start(self):
        """adds data to self.data until termination signal"""
        self.initiate()
        print(" -- starting stream")
        self.keepRecording = True  # set this to False later to terminate stream
        self.data = None  # will fill up with threaded recording data
        self.fft = None
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1,
                                  rate=self.rate, input=True, frames_per_buffer=self.chunk)
        self.stream_thread_new()
        self.stream_thread_onset()

    def rt_onset(self):
        self.start = time.time()
        o_step = 1024
        o_win_len = o_step * 2
        hlf_win = np.int(o_win_len / 2)
        prev_data = []
        current_data = []
        time.sleep(0.1)
        self.Start = time.time()
        for i in range(1, int(round(self.samples / self.chunk))):
            prev_data = np.append(prev_data, self.data)
            time.sleep(self.chunk / self.rate)

        theta1 = np.zeros(hlf_win)
        theta2 = theta1
        oldmag = theta1
        df = []
        ts = []
        timdif = 0
        endn = self.Start
        count = 0
        xold = 0
        while self.TV:
            begn = time.time()
            current_data = self.data
            temp, theta1, theta2, oldmag = onset_timer.onset_detection(np.array(np.append(prev_data, current_data)),
                                                                       xold,
                                                                       theta1, theta2, oldmag, self.rate)
            progend = time.time()
            df = df + [temp]
            ts = ts + [endn]
            xold = current_data[len(current_data) - 1]
            try:
                time.sleep(self.chunk / self.rate - progend + begn)
            except:
                count = count + 1
                print(count)
                print('Warning: Algo took too Long...!!!')
            prev_data = np.append(prev_data[self.chunk:len(prev_data)], current_data)

            if (len(df) >= self.adaptval):
                if (len(self.tdf) >= self.lentdf):
                    self.tdf = self.tdf[1:]
                    self.time_stamps = self.time_stamps[1:]
                dfnew = np.array(onset_timer.part_adapt_thresh(df))
                if (len(self.tdf) == 0):
                    self.tdf = dfnew
                    self.time_stamps = ts
                    self.assigncost(dfnew)
                else:
                    self.tdf = np.append(self.tdf, [dfnew[len(dfnew) - 1]], axis=0)
                    self.time_stamps = np.append(self.time_stamps, [ts[len(ts) - 1]], axis=0)
                    self.assigncost(dfnew[len(dfnew) - 1])
                df = df[1:]
                ts = ts[1:]

                if (len(self.tdf) >= self.acorrwin and len(self.tdf)%15==0):
                    self.pmax = np.round(60 / 65 * (self.fs / self.conver))
                    self.pmin = np.round(60 / 130 * (self.fs / self.conver))
                    temp = self.getbpm()
                    temp = np.array(onset_timer.adapt_threshold(list(temp)))
                    self.abpm = np.concatenate((self.abpm, temp), axis=0)
                    temp = np.argmax(temp)
                    self.bepm = self.bepm + [temp]
                    self.bpm = np.median(self.bepm[min(0, len(self.bepm) - 10):])
                    if(self.bepmdebug!=[]):
                        if(abs(self.bpm-self.bepmdebug[len(self.bepmdebug)-1])>self.bpm*self.thresh):
                            self.C=[]
                    self.bepmdebug = self.bepmdebug + [self.bpm]


            endn = time.time()
            timdif = endn - begn
            if timdif != 0:
                self.conver = 1024

    def assigncost(self, df):
        if np.size(self.bpm) != 0 and self.bpm != 0:
            if (len(self.C) >= self.lentdf):
                self.C = self.C[1:]
            for j in range(0, np.size(df)):
                maxcost = float('-inf')
                Start = int(round(len(self.tdf) - self.bpm - self.thresh * self.bpm))
                Stop = int(round(len(self.tdf) - self.bpm + self.thresh * self.bpm + 1))
                for i in range(Start, Stop):
                    if (i < 0):
                        if (np.size(df) == 1):
                            cost = df
                        else:
                            cost = df[j]
                    else:
                        if (i > len(self.C) - 1):
                            self.C = self.tdf
                            return
                        if (np.size(df) == 1):
                            cost = 0.9 * self.C[i] + df
                        else:
                            cost = 0.9 * self.C[i] + df[j]
                    if cost > maxcost:
                        maxcost = cost
                self.C = np.append(self.C, [maxcost], axis=0)
            if (self.firstrunflag == 1):
                self.lastbeat = int(round(len(self.C) - self.bpm - 1)) + np.argmax(
                    self.C[int(round(len(self.C) - 1 - self.bpm)):len(self.C) - 1])
                self.lbdebug = self.time_stamps[self.lastbeat] - self.Start
                self.stream_thread_start_beatseq()
                self.firstrunflag = 0
            else:
                oldbeat = self.lastbeat
                try:
                    part_array = self.C[int(round(self.lastbeat + self.bpm * (1 - self.thresh))):int(
                        round(self.lastbeat + self.bpm * (1 + self.thresh))) + 1]
                    if (len(part_array) == (int(round(self.lastbeat + self.bpm * (1 + self.thresh))) + 1 - int(
                            round(self.lastbeat + self.bpm * (1 - self.thresh))))):
                        self.lastbeat = int(round(self.lastbeat + self.bpm * (1 - self.thresh))) + np.argmax(part_array)
                        self.lbdebug = np.append(self.lbdebug, self.time_stamps[self.lastbeat] - self.Start)
                        self.stream_thread_start_beatseq()
                except:
                    self.lastbeat = self.lastbeat

    def printbeat(self):
        present_time = time.time()
        try:
            time.sleep(self.bpm * self.conver / self.rate - (present_time - self.time_stamps[self.lastbeat]))
            self.BT = np.append(self.BT, [time.time() - self.Start], axis=0)
            self.stream_gpio_led(100)
#            print('Beat')            
			#playbeep.playbeep(self.fs,1000,0.15)
            #time.sleep(0.15)
        except:
            time.sleep(0.001)


    def gpio_led(self,x=0.1):
        GPIO.output(7, GPIO.HIGH)
        time.sleep(x)
        GPIO.output(7, GPIO.LOW)

    def getbpm(self):
        n = np.arange(1, self.acorrwin)
        wv = (
        np.divide(n, np.power(self.rayparam, 2)) * np.exp(-np.divide(np.power(n, 2), (2 * np.power(self.rayparam, 2)))))
        eps = np.finfo(float).eps
        wv = wv / np.sum(eps + wv)
        if (len(self.tdf) >= self.acorrmaxwin):
            n = np.arange(1, self.acorrmaxwin)
            wv = (np.divide(n, np.power(self.rayparam, 2)) * np.exp(
                -np.divide(np.power(n, 2), (2 * np.power(self.rayparam, 2)))))
            eps = np.finfo(float).eps
            wv = wv / np.sum(eps + wv)
            a = self.tdf[int(round(len(self.tdf) - self.acorrmaxwin + 1)):]
        else:
            a = self.tdf[int(round(len(self.tdf) - self.acorrwin + 1)):]
        self.acorr = np.correlate(a, a, "full")
        self.acorr = self.acorr[np.int(len(self.acorr) / 2):]
        self.acorr = np.append(0, self.acorr)
        rcf = np.zeros(len(self.acorr))
        numelem = 4
        for i in range(int(self.pmin), int(self.pmax)):  # maximum beat period
            for a in range(1, numelem + 1):  # number of comb elements
                for b in range(1 - a, a):  # gs using normalization of comb elements
                    if ((a * i + b) < len(self.acorr)):
                        rcf[i] = rcf[i] + (self.acorr[a * i + b] * wv[i]) / (2 * a - 1)
        return rcf

if __name__=="__main__":
    ear = BeatTrackRT()
    song_no=input('Enter Song Number :')
    ear.stream_start()  # goes forever
    try:
        os.mkdir('Beats')
    except:
        pass
    while ear.TV:
        time.sleep(0.001)
        typedString = input()
        if typedString =='f' or typedString =='S':
            f4 = open('Beats/'+song_no+'.txt', 'w')
            f4.write("".join(str(x) + "\n" for x in ear.BT))  # python will convert \n to os.linesep
            f4.close()
            print("Written to file!")
        if typedString =='i' or typedString =='S':
            stop=time.time()
            print(len(ear.tdf))
            ear.TV=False
            print(stop-ear.start)
            print("Interrupted By ME :P")
            ear.close()
    print("DONE")
    # Enter song no and start playing song at the same time all the above data is saved when  u press S
