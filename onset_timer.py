import numpy as np

#rpi_gpio_out = 0
# lbo = 0
# bri = 0
# initial_waiting_time = 2 #in seconds
#x = np.zeros(fs*initial_waiting_time)


# def master_timer(lbo, bri):
# 	global rpi_gpio_out
# 	rpi_gpio_out = 0
# 	time.sleep(bri)
# 	rpi_gpio_out = 1

def matlab_buffer(list,grp,overlap):
	z=(grp-len(list)%(grp-overlap))%grp
	list=list+[0]*z
	return [list[i:i+grp] for i in range(0,len(list)-overlap,grp-overlap)]

def adapt_threshold(df,pre=8,post=7):
	N=len(df)
	m=[]
	for i in range(0,min(post, N)):
		k = min(i + pre, N)
		m = np.append(m,np.mean(df[0:k]))

	if N > (post + pre):
		m = np.append(m,np.mean(np.array(matlab_buffer(df, post + pre + 1, post + pre)),axis=1))

	for i in range(N-pre,N):
		j = max(i - post, 1)-1
		m = np.append(m,np.mean(df[j:len(df)-1]))
	df1=np.zeros(len(df))
	np.subtract(np.array(df),np.array(m),df1)
	return (df1>0)*df1


def onset_detection(x, xold, theta1, theta2, oldmag, fs=44100):
	x=np.append(xold,x)
	x=np.diff(x)
	o_step  	= 1024
	o_win_len   = o_step * 2
	win_hann 	= np.hanning(o_win_len)
	df = []
	x_fft	= np.fft.fft(win_hann*x)
	x_fft	= x_fft[0:np.int(len(x_fft)/2)]
	mag   = (np.absolute(x_fft))
	theta = np.angle(x_fft)
	dev   = ((theta-2*theta1+theta2 + np.pi) %  (-2 * np.pi)) + np.pi
	meas  = oldmag - (mag*np.exp(1j* dev))
	df = np.sum(np.sqrt(np.power((np.real(meas)),2) + np.power((np.imag(meas)),2)))
	return (df, theta, theta1, mag)

def part_adapt_thresh(inp):
	m=np.mean(inp);
	inp=inp-m
	out=inp*(inp>0)
	return out




