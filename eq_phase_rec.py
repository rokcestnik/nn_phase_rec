import random
from math import pi, floor, sqrt


###################################################
#####   PHASE RECONSTRUCTION FROM EQUATIONS   #####
###################################################


def one_step_integrator(state, ders, ps, dt):
	"""RK4 integrates state with derivative and input for one step of dt
	
	:param state: state of the variables
	:param ders: derivative functions
	:param ps: perturbations [p1,p2,...] (default 0)
	:param dt: time step
	:return: state after one integration step"""
	D = len(state)
	# 1
	k1 = [ders[i](state,ps) for i in range(D)]
	# 2
	state2 = [state[i]+k1[i]*dt/2.0 for i in range(D)]
	k2 = [ders[i](state2,ps) for i in range(D)]
	# 3
	state3 = [state[i]+k2[i]*dt/2.0 for i in range(D)] 
	k3 = [ders[i](state3,ps) for i in range(D)]
	# 4
	state4 = [state[i]+k3[i]*dt for i in range(D)] 
	k4 = [ders[i](state4,ps) for i in range(D)]
	# put togeather
	statef = [state[i] + (k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0*dt for i in range(D)]
	return statef


def generate_signal(ders, n, sampling, initial_state=None, number_of_variables=1, number_of_perturbations=1, warmup_time=1000.0, tau=3.0, eps=0.5, dt=0.01):
	"""generates signal for the oscillator driven by correlated noise from dynamical equations
	
	:param ders: a list of state variable derivatives
	:param n: length of time series
	:param sampling: sampling rate
	:param initial_state: initial state
	:param number_of_variables: number of variables returned, not including the input (default 1)
	:param number_of_perturbations: number of perturbations (default 1)
	:param warmup_time: time for relaxing to the stationary regime (default 1000)
	:param tau: noise correlation time (default 3.0)
	:param eps: noise strength (default 0.5)
	:param dt: time step (default 0.01)
	:return: time series of the signal and driving noise"""
	# initial conditions
	if(initial_state==None):
		state = [random.gauss(0,1) for i in range(len(ders))]
	else:
		state = initial_state
	res_s = [[] for i in range(number_of_variables)] # resulting signal
	res_p = [[] for j in range(number_of_perturbations)] # resulting perturbations
	ps = [0 for p in range(number_of_perturbations)]
	# warmup
	for i in range(round(warmup_time/dt)):
		for p in range(number_of_perturbations):
			ps[p] = ps[p] - (ps[p]/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt 
		state = one_step_integrator(state, ders, ps, dt)
	# real integration
	for i in range(n*sampling):
		for p in range(number_of_perturbations):
			ps[p] = ps[p] - (ps[p]/tau - eps*sqrt(2/tau)*random.gauss(0,1)/sqrt(dt))*dt 
		state = one_step_integrator(state, ders, ps, dt)
		for c in range(number_of_variables):
			res_s[c].append(state[c])
		for p in range(number_of_perturbations):
			res_p[p].append(ps[p])
	# sampling
	res_s = [res_s[i][::sampling] for i in range(number_of_variables)]
	res_p = [res_p[i][::sampling] for i in range(number_of_perturbations)]
	return res_s, res_p


def oscillator_period(ders, inp=[0 for i in range(10)], initial_state=None, warmup_time=1500.0, thr=0.0, dt=0.01):
	"""calculates the natural period of the oscillator from dynamical equations
	
	:param ders: a list of state variable derivatives
	:param inp: vector of offset inputs to the system (default 0 vector)
	:param initial_state: intial state (default None)
	:param warmup_time: time for relaxing to the stable orbit (default 1000)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:return: natural period"""
	# initial conditions
	if(initial_state==None):
		state = [random.gauss(0,1) for i in range(len(ders))]
	else:
		state = initial_state
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp, dt)
	# integration up to x = thr
	xh = state[0]
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
	# Henon trick
	dt_beggining = 1.0/ders[0](state,inp)*(state[0]-thr)
	# spoil condition and go again to x = 0 (still counting time)
	xh = state[0]
	time = 0
	while((state[0] > thr and xh < thr) == False):
		xh = state[0]
		state = one_step_integrator(state, ders, inp, dt)
		time = time + dt
	# another Henon trick
	dt_end = 1.0/ders[0](state,inp)*(state[0]-thr)
	return time + dt_beggining - dt_end


def oscillator_PRC(ders, direction, inp=[0 for i in range(10)], initial_state=None, warmup_time=1000.0, stimulation=1.0, period_counts=5, dph=0.1, thr=0.0, dt=0.01):
	"""calculates the phase response curve from dynamical equations
	
	:param ders: a list of state variable derivatives
	:param direction: direction in which the response is probed
	:param inp: vector of offset inputs to the system (default 0 vector)
	:param initial_state: intial state (default None)
	:param warmup_time: time for relaxing to the stable orbit (default 1000)
	:param stimulation: strength of the stimulation (default 1.0)
	:param period_counts: how many periods to wait for evaluating the asymptotic phase shift (default 5)
	:param dph: phase resolution (default 0.1)
	:param thr: threshold for determining period (default 0.0)
	:param dt: time step (default 0.01)
	:return: the phase response curve"""
	period = oscillator_period(ders, inp)
	PRC = [[dph*i for i in range(floor(2*pi/dph))],[0 for i in range(floor(2*pi/dph))]] # PRC list
	# initial conditions
	if(initial_state==None):
		state = [random.gauss(0,1) for i in range(len(ders))]
	else:
		state = initial_state
	# normalize the direction
	norm = sqrt(sum(direction[i]**2 for i in range(len(direction))))
	direction = [direction[i]/norm for i in range(len(direction))]
	# warmup
	for i in range(round(warmup_time/dt)):
		state = one_step_integrator(state, ders, inp, dt)
	# stimulating phases
	for ph in [dph*i for i in range(floor(2*pi/dph))]:
		# integration up to x = thr
		xh = state[0]
		while((state[0] > thr and xh < thr) == False):
			xh = state[0]
			state = one_step_integrator(state, ders, inp, dt)
		# Henon trick
		dt_beggining = 1.0/ders[0](state,inp)*(state[0]-thr)
		# spoil condition and go to ph (counting time)
		xh = state[0]
		time = dt_beggining
		while(time < ph/(2*pi)*period):
			xh = state[0]
			state = one_step_integrator(state, ders, inp, dt)
			time = time + dt
		# stimulate 
		xh = state[0]
		state_stim = [state[i]+direction[i]*dt for i in range(len(state))] # shift the state
		state = one_step_integrator(state_stim, ders, inp, dt)
		time = time + dt
		#integrate for some periods
		for p in range(period_counts):
			xh = state[0] # spoil condition
			while((state[0] > thr and xh < thr) == False):
				xh = state[0]
				state = one_step_integrator(state, ders, inp, dt)
				time = time + dt
		# another Henon trick
		dt_end = 1.0/ders[0](state,inp)*(state[0]-thr)
		phase_shift = 2*pi*(period_counts*period - (time-dt_end))/period
		PRC[1][round(ph/dph)] = phase_shift/(stimulation*dt)
	return PRC
