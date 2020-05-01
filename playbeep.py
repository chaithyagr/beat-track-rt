import math
import pyaudio
def playbeep(RATE=44100,WAVE=1000, LEN=0.1):
    LEN=round(LEN*RATE)
    PyAudio = pyaudio.PyAudio
    data = ''.join([chr(int(math.sin(x/((RATE/WAVE)/math.pi))*127+128)) for x in range(1,LEN)])
    p = PyAudio()
    stream = p.open(format =
                p.get_format_from_width(1),
                channels = 1,
                rate = RATE,
                output = True)
    for DISCARD in range(1,2):
        stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
