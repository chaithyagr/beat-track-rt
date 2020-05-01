# IEEE SP Cup 2017
## Real Time Beat Tracking

### Requirements

* Python 3.x

### Packages Required
* PyAudio
* Numpy
* Glob
* Wave
* threading
* os
* Math
* time
* RPi.GPIO (for Raspberry Pi 3)

### Operating System used:
* Debian Linux
* UbuntuMate (for Raspberry Pi 3)
	
## Codes

**BeatTrackAll**

This code runs a batch of songs. When the code is executed, the code prompts for the directory where the song pieces are present. Before running this code, please bridge your audio input from speaker to microphone. The code will play the song sequentially and track beats. The predicted beats are saved in a file in the folder, 'Beat' in the same directory where the code is present. Inside 'Beat', files are created with numbers based on the sequence in which the code runs the song pieces.

**BeatTrackRT**

This code must be run on Raspberry Pi 3. This code tracks beats for a song piece played through a microphone. Please use external sound card and configure the microphone appropriately through the OS before running the code. The code prompts for song number which is the filename of output file with beats. Also whenever beat occurs, the Raspberry Pi gives a Digital High on pin 7. Please Press S (Capital) to save beats and stop the code. If you want to quit without saving beats press i. Please note that you will have to restart the code for each song piece so that cost function is cleared and predicted beats have good accuracy.



