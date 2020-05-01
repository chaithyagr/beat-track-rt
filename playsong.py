def playsong(self):
    wf = wave.open(self.filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(self.chunk)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(self.chunk)
