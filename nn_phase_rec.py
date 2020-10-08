from math import sin, cos, pi, sqrt, floor, ceil, atan
import numpy as np


############################################################
#####   PHASE RECONSTRUCTION USING NEAREST NEIGHBORS   #####
############################################################


def nn_one_step_integrate_linear(data, pert_data, state, neighbors, evolution_step=1):
	"""integrate using the linear approximation for f (for system \dot{x} = f(x) + p)
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param state: state of the system, [x,y,z...]
	:param neighbors: how many closest neighbors are considered
	:param evolution_step: by how many steps it is evolved at once, meant for no_input prediction, not accurate here unless the perturbation is slow (default 1)
	:return: new time-evolved state """
	# dimensions
	n = len(data)
	m = len(pert_data)
	# check if there are enough data points
	min_points = (n+1)+m + 1
	if(neighbors < min_points):
		neighbors = min_points
	# just transpose the data
	states_data = np.array(data).transpose()
	p_states_data = np.array(pert_data).transpose()
	# nearest neighbors
	closest_index = nearest_neighbors(states_data, state, neighbors, last_index_offset=evolution_step)
	# now lets setup the optimization
	A = [] # the matrix
	der_state = [[] for i in range(n)]
	for i in range(neighbors):
		# A line
		Aline = []
		Aline.append(1) # constant
		for j in range(n):
			Aline.append(states_data[closest_index[i]][j]) # (x,y,z...)
		for j in range(m):
			Aline.append(p_states_data[closest_index[i]][j]) # p_j
		# append to A and difference vectors
		A.append(Aline)
		for j in range(n):
			der_state[j].append(states_data[closest_index[i]+evolution_step][j]-states_data[closest_index[i]][j])
	# the minimization
	A = np.matrix(A)
	AT = A.transpose()
	ATA = AT*A
	ATAI = np.linalg.inv(ATA)
	ATAIAT = ATAI*AT
	# the resulting linear fit coefficients
	coeffs = [ATAIAT*np.matrix(der_state[i]).transpose() for i in range(n)]
	return [state[i] + float(coeffs[i][0]) + sum(float(coeffs[i][1+j])*state[j] for j in range(n)) for i in range(n)] # in general there would be another term for the perturbation (but here we take the unperturbed evolution)


def nn_one_step_integrate_constant(data, pert_data, state, neighbors, evolution_step=1):
	"""integrate using the constant approximation for f (for system \dot{x} = f(x) + p)
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param state: state of the system, [x,y,z...]
	:param neighbors: how many closest neighbors are considered
	:param evolution_step: by how many steps it is evolved at once, meant for no_input prediction, not accurate here unless the perturbation is slow (default 1)
	:return: new time-evolved state """
	# dimensions
	n = len(data)
	m = len(pert_data)
	# check if there are enough data points
	min_points = 1+m + 1
	if(neighbors < min_points):
		neighbors = min_points
	# just transpose the data
	states_data = np.array(data).transpose()
	p_states_data = np.array(pert_data).transpose()
	# nearest neighbors
	closest_index = nearest_neighbors(states_data, state, neighbors, last_index_offset=evolution_step)
	# now lets setup the optimization
	A = [] # the matrix
	der_state = [[] for i in range(n)]
	for i in range(neighbors):
		# A line
		Aline = []
		Aline.append(1) # constant
		for j in range(m):
			Aline.append(p_states_data[closest_index[i]][j]) # p_j
		# append to A and difference vectors
		A.append(Aline)
		for j in range(n):
			der_state[j].append(states_data[closest_index[i]+evolution_step][j]-states_data[closest_index[i]][j])
	# the minimization
	A = np.matrix(A)
	AT = A.transpose()
	ATA = AT*A
	ATAI = np.linalg.inv(ATA)
	ATAIAT = ATAI*AT
	# the resulting linear fit coefficients
	coeffs = [ATAIAT*np.matrix(der_state[i]).transpose() for i in range(n)]
	return [state[i] + float(coeffs[i][0]) for i in range(n)] # in general there would be another term for the perturbation (but here we take the unperturbed evolution)


def nn_one_step_no_input(data, pert_data, state, neighbors, evolution_step=1):
	"""integrate using only the averaging (no perturbation)
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are p_x(t) (for now I'm only taking perturbations in x direction)
	:param pert_data: perturbation data (not used here, but needed because this function gets called in the same way than integrations that need it)
	:param state: state of the system, [x,y,z,...]
	:param neighbors: how many closest neighbours are considered
	:param evolution_step: by how many steps it is evolved at once (default 1)
	:return: new time-evolved state """	
	# just transpose the data
	states_data = np.array(data).transpose()
	# nearest neighbors
	closest_index = nearest_neighbors(states_data, state, neighbors, last_index_offset=evolution_step)
	where_they_go = [[data[j][closest_index[i]+evolution_step] + state[j]-data[j][closest_index[i]] for j in range(len(data))] for i in range(neighbors)] # here are the points in the next step, accounted for their relative position towards the original point
	points = [[where_they_go[i][j] for i in range(neighbors)] for j in range(len(data))]
	# just average them
	new_point = [sum(points[j][i] for i in range(neighbors))/neighbors for j in range(len(data))]
	return new_point


def nn_integrate(data, pert_data, n, initial_state, neighbors, expansion='const', evolution_step=1):
	"""integrate a trajectory using nearest neighbor prediction
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param n: number of steps
	:param initial_state: initial state of the system, [x,y,z,...]
	:param neighbors: how many closest neighbours are considered
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: timestep length in units of data timesteps (default 1)
	:return: nn integrated trajectory"""
	print("nearest neighbor integration...")
	# set the integration function
	int_function = set_integration_function(expansion)
	trajectory = [initial_state]
	state = initial_state.copy()
	for p in range(n):
		if(p%100 == 0): 
			print("\t", p, "/", n)
		# integrate
		state = int_function(data, pert_data, state, neighbors, evolution_step)
		# save state
		trajectory.append(state)
	return trajectory


def nn_limit_cycle(data, pert_data, T0_arg, neighbors, initial_state=None, warmup_steps=2500, expansion='const', evolution_step=1):
	"""estimate the limit cycle with the nearest neighbor prediction
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param neighbors: how many closest neighbours are considered
	:param initial_state: initial state of the system, [x,y,z,...] (dafault the first state from the data)
	:param warmup_steps: warmup timesteps to reach the limit cycle (default 2500)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: timestep length in units of data timesteps
	:param evolution_step: the evolution_step parameter used in nn_integration function for phase determination (default 1)
	:return: limit cycle as an array of states [[x0,y0,z0,...],[x1,y1,z1,...],...]
	"""
	print("getting the nn limit cycle...")
	# express the natural period in effective steps
	T0 = T0_arg/evolution_step
	# set the integration function
	int_function = set_integration_function(expansion)
	# initial conditions
	if(initial_state==None):
		state = [data[i][0] for i in range(len(data))]
	else:
		state = initial_state
	# warmup
	for i in range(warmup_steps):
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# number of points
	N = ceil(T0)+1
	# set the delta_t
	delta_t = T0/N
	# limit cycle
	lc = [state]
	for i in range(N):
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
		lc.append(state)
	return lc


def nn_period(data, pert_data, thr, neighbors, initial_state=None, warmup_steps=2500, expansion='const', period_count_avg=1, evolution_step=1):
	"""estimate the natural period with the nearest neighbor prediction (in units of data steps, data[i][j] == data[i+n*T0][j])
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param neighbors: how many closest neighbours are considered
	:param initial_state: initial state of the system, [x,y,z,...] (dafault the first state from the data)
	:param warmup_steps: warmup timesteps to reach the limit cycle (default 2500)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param period_count_avg: number of periods the period is averaged over (default 1)
	:param evolution_step: the evolution_step parameter used in nn_integration function (default 1)
	:return: time of one period in timesteps of the data
	"""
	print("nn measuring the period...")
	# set the integration function
	int_function = set_integration_function(expansion)
	# initial conditions
	if(initial_state==None):
		state = [data[i][0] for i in range(len(data))]
	else:
		state = initial_state
	# warmup
	for i in range(warmup_steps):
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# threshold crossing
	x_prev = state[0]
	while( not (x_prev < thr and state[0] > thr)):
		x_prev = state[0]
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# T0 is already something because the current state is over the threshold
	T0 = (state[0]-thr)/(state[0]-x_prev)
	# integrate until you cross the threshold 'period_count_avg' times again
	for pc in range(period_count_avg):
		# break the condition
		x_prev = state[0]
		while( not (x_prev < thr and state[0] > thr)):
			x_prev = state[0]
			state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
			T0 = T0+1
	# substract for how much it went over
	T0 = T0 - (state[0]-thr)/(state[0]-x_prev)
	return T0/period_count_avg*evolution_step


def nn_phase(data, pert_data, thr, T0_arg, state, neighbors, warmup_periods=1, expansion='const', evolution_step=1):
	"""estimate the asymptotic phase with the nearest neighbor prediction
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param state: state of the system, [x,y,z,...]
	:param neighbors: how many closest neighbours are considered
	:param warmup_periods: warmup periods to reach the limit cycle (default 1)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function (default 1)
	:return: asymptotic phase of the state
	"""
	print("nn measuring the phase...")
	# express the natural period in effective steps
	T0 = T0_arg/evolution_step
	# set the integration function
	int_function = set_integration_function(expansion)
	# integrate for 'warmup_periods' periods to get to the limit cycle
	for i in range(ceil(warmup_periods*T0)):
		x_prev = state[0]
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	T = ceil(T0)-T0
	# count the time until reaching zero phase (threshold crossing)
	while( not (x_prev < thr and state[0] > thr)):
		x_prev = state[0]
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
		T = T+1
	# correct for the threshold overcrossing
	T = T - (state[0]-thr)/(state[0]-x_prev)
	return (T0-T)/T0


def nn_prc_in_point(state, data, pert_data, thr, T0_arg, direction, neighbors, perturbation=0.1, expansion='const', evolution_step=1):
	"""estimate the phase response in one point
	
	:param state: state [x,y,z,...] in which the phase response is evaluated
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param direction: direction in which the response is probed
	:param neighbors: how many closest neighbours are considered
	:param perturbation: amplitude of the perturbation (default 0.1)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function (default 1)
	:return: phase response at the designated state
	"""
	state_shift = [state[i]+direction[i]*perturbation for i in range(len(state))]
	phase = nn_phase(data, pert_data, thr, T0_arg, state, neighbors, expansion=expansion, evolution_step=evolution_step)
	phase_s = nn_phase(data, pert_data, thr, T0_arg, state_shift, neighbors, expansion=expansion, evolution_step=evolution_step)
	return (phase_s-phase)/perturbation*(2*pi)


def nn_avg_prc_in_point(state, data, pert_data, thr, T0_arg, direction, neighbors, perturbation_range=0.15, perturbation_increment=0.005, expansion='const', evolution_step=1):
	"""estimate the average phase response in one point
	
	:param state: state [x,y,z,...] in which the phase response is evaluated
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param direction: direction in which the response is probed
	:param neighbors: how many closest neighbours are considered
	:param perturbation_range: maximal amplitude of the perturbation (default 0.15)
	:param perturbation_increment: perturbation increment (default 0.005)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function (default 1)
	:return: phase response at the designated state
	"""
	# evaluate in how many points to estimate phase
	n_per_side = int(floor(perturbation_range/perturbation_increment))
	n = 2*n_per_side+1
	deltas = [(-n_per_side+i)*perturbation_increment for i in range(n)]
	# estimate the phases
	phases = [nn_phase(data, pert_data, thr, T0_arg, np.array(state)+np.array(direction)*deltas[i], neighbors, warmup_periods=1, expansion=expansion, evolution_step=evolution_step) for i in range(n)]
	# prepare the matrix for fitting the line
	A = np.matrix([deltas,[1 for i in range(len(deltas))]]).transpose()
	ATAA = np.linalg.inv(A.transpose()*A)*A.transpose()
	return 2*pi*ATAA.dot(phases)[0,0]


def nn_prc(data, pert_data, thr, T0_arg, direction, neighbors, initial_state=None, perturbation=0.1, sampling=10, expansion='const', evolution_step=1):
	"""measures the phase response curve with respect to perturbation direction
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param direction: direction in which the response is probed
	:param neighbors: how many closest neighbours are considered
	:param initial_state: initial state of the system, [x,y,z,...] (dafault the first state from the data)
	:param perturbation: amplitude of the perturbation (default 0.1)
	:param sampling: phase sampling of the phase response curve (default 10)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function for phase determination (default 1)
	:return: phase response curve as an array [[ph1,response1], [ph2,response2],...]
	"""
	print("getting the nn prc...")
	equi_points = nn_equiphase_points(data, pert_data, thr, T0_arg, neighbors, initial_state=initial_state, sampling=sampling, expansion=expansion, evolution_step=evolution_step)
	prc = []
	for p in range(sampling):
		prc.append([2*pi*(0.5+p)/sampling, nn_prc_in_point(equi_points[p], data, pert_data, thr, T0_arg, direction, neighbors, perturbation=perturbation, expansion=expansion, evolution_step=evolution_step)])
	return prc


def nn_avg_prc(data, pert_data, thr, T0_arg, direction, neighbors, initial_state=None, perturbation_range=0.15, perturbation_increment=0.005, sampling=10, expansion='const', evolution_step=1):
	"""measures the average phase response curve with respect to perturbation direction
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param direction: direction in which the response is probed
	:param neighbors: how many closest neighbours are considered
	:param initial_state: initial state of the system, [x,y,z,...] (dafault the first state from the data)
	:param perturbation_range: maximal amplitude of the perturbation (default 0.15)
	:param perturbation_increment: perturbation increment (default 0.005)
	:param sampling: phase sampling of the phase response curve (default 10)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function for phase determination (default 1)
	:return: average phase response curve as an array [[ph1,response1], [ph2,response2],...]
	"""
	print("getting the nn avg prc...")
	equi_points = nn_equiphase_points(data, pert_data, thr, T0_arg, neighbors, initial_state=initial_state, sampling=sampling, expansion=expansion, evolution_step=evolution_step)
	prc = []
	for p in range(sampling):
		prc.append([2*pi*(0.5+p)/sampling, nn_avg_prc_in_point(equi_points[p], data, pert_data, thr, T0_arg, direction, neighbors, perturbation_range=perturbation_range, perturbation_increment=perturbation_increment, expansion=expansion, evolution_step=evolution_step)])
	return prc


def nn_iso(data, pert_data, thr, phase, T0_arg, neighbors, initial_state=None, phase_warmup_periods=1, expansion='const', evolution_step=1):
	"""approximate an isochron corresponding to phase 'phase' with the nearest neighbor prediction (2D only)
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param phase: phase of the isochron
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param neighbors: how many closest neighbours are considered
	:param initial_state: initial state of the system, [x,y,z,...] (dafault the first state from the data)
	:param phase_warmup_periods: how many periods is the state let to relax to the limit cycle when evaluating the asymptotic phase (default 1)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function for phase determination (default 1)
	:return: isochron as an array [isos_in, isos_out] where isos_in and isos_out are arrays [[x0,y0,z0,...],[x1,y1,z1,...],...] corresponding to points inside and outside the limit cycle respectively"""
	print("getting the nn isochrons...")
	# express the natural period in effective steps
	T0 = T0_arg/evolution_step
	# set the integration function
	int_function = set_integration_function(expansion)
	# initial conditions
	if(initial_state==None):
		state = [data[i][0] for i in range(len(data))]
	else:
		state = initial_state
	# integrate for 3 periods to get to the limit cycle
	for i in range(3*ceil(T0)):
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# get the approximate center of the limit cycle
	xavg = 0
	yavg = 0
	for i in range(ceil(T0)):
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
		xavg = xavg + state[0]
		yavg = yavg + state[1]
	xavg = xavg/ceil(T0)
	yavg = yavg/ceil(T0)
	# reach zero phase according to the threshold
	x_prev = state[0]
	while( not (x_prev < thr and state[0] > thr)):
		x_prev = state[0]
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# correct the start time for the threshold overcrossing
	T = (state[0]-thr)/(state[0]-x_prev)
	# count time until reaching the desired phase
	while( T/T0 < phase ):
		previous_state = state.copy()
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
		T = T+1
	# correct for the overcrossing
	dT = T0*(T/T0-phase)
	state[0] = previous_state[0] + (1-dT)*(state[0]-previous_state[0])
	state[1] = previous_state[1] + (1-dT)*(state[1]-previous_state[1])
	# measure the phase of the starting point (ideally it should be the same as the argument 'phase')
	start_phase = nn_phase(data, pert_data, thr, T0_arg, state, neighbors, warmup_periods=phase_warmup_periods, expansion=expansion, evolution_step=evolution_step)
	# starting direction
	direction = [(state[0]-xavg)/36, (state[1]-yavg)/36] # move in steps of sqrt(1/36^2+1/36^2) = sqrt(2)/36 (a bit arbitrary, should change for some systems)
	# UNTIL HERE WE WERE JUST GETTING THE STARTING POINTS READY
	# measure the phase in the left direction
	direction_left = rotate(direction, pi/2)
	point_left = [state[0]+direction_left[0], state[1]+direction_left[1]]
	phase_left = nn_phase(data, pert_data, thr, T0_arg, point_left, neighbors, warmup_periods=phase_warmup_periods, expansion=expansion, evolution_step=evolution_step)
	# measure the phase in the right direction
	direction_right = rotate(direction, -pi/2)
	point_right = [state[0]+direction_right[0], state[1]+direction_right[1]]
	phase_right = nn_phase(data, pert_data, thr, T0_arg, point_right, neighbors, warmup_periods=phase_warmup_periods, expansion=expansion, evolution_step=evolution_step)
	# determine which direction is the phase increasing
	phase_dir = 1
	if(phase_left < phase_right):
		phase_dir = -1
	# UNTIL HERE WE SETUP THE BIJECTION AND MEASURE IN WHICH DIRECTION DOES THE PHASE INCREASE
	point = state.copy()
	isos_out = [point]
	# loop over how many points
	for p in range(10):
		print("\tp = ", p, "/ 10")
		# iterate
		interval = pi/4 # starting direction correction interval (it can be corrected by at most pi/2 after infinite bijection)
		for it in range(5):
			# measure the phase one step in the same direction
			point_front = [point[0]+direction[0], point[1]+direction[1]]
			phase_front = nn_phase(data, pert_data, thr, T0_arg, point_front, neighbors, warmup_periods=phase_warmup_periods, expansion=expansion, evolution_step=evolution_step)
			# rotate in the right direction
			direction = rotate(direction, sgn(start_phase-phase_front)*interval*phase_dir)
			# divide the interval in half
			interval = interval/2
		# new point
		point = [point[0]+direction[0], point[1]+direction[1]]
		isos_out.append(point)
	# UNTIL HERE WAS THE CALCULATION FOR THE ISOCHRON OUTSIDE THE LIMIT CYCLE, NOW REPEAT FOR INSIDE
	# starting direction
	direction = [-(state[0]-xavg)/36, -(state[1]-yavg)/36] # move in steps of sqrt(1/36^2+1/36^2) = sqrt(2)/36 (a bit arbitrary, should change for some systems)
	# the phase directions is opposite to what is outside
	phase_dir = -phase_dir
	# and now again bijection
	point = state.copy()
	isos_in = [point]
	# loop over how many points
	for p in range(10):
		print("\tp = ", p, "/ 10")
		# iterate
		interval = pi/4 # starting direction correction interval (it can be corrected by at most pi/2 after infinite bijection)
		for it in range(5):
			# measure the phase one step in the same direction
			point_front = [point[0]+direction[0], point[1]+direction[1]]
			phase_front = nn_phase(data, pert_data, thr, T0_arg, point_front, neighbors, warmup_periods=phase_warmup_periods, expansion=expansion, evolution_step=evolution_step)
			# rotate in the right direction
			direction = rotate(direction, sgn(start_phase-phase_front)*interval*phase_dir)
			# divide the interval in half
			interval = interval/2
		# new point
		point = [point[0]+direction[0], point[1]+direction[1]]
		isos_in.append(point)
	return [isos_in, isos_out]



###################################
#####   AUXILIARY FUNCTIONS   #####
###################################


def nearest_neighbors(data, state, neighbors, same_trajectory_proximity_ban=8, last_index_offset=1): 
	""" find the nearest neighbors
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param state: state [x,y,z,...] of which nearest neighbors we are looking for
	:param neighbors: how many closest neighbors are considered
	:param same_trajectory_proximity_ban: there can be only one point within the time of this parameter, not allowing points too close in time to count as independent neighbors
	:param last_index_offset: in checking for neighbors the very last point(s) should not be taken into account because in the integration function that follows the succeeding points are called (default 1)
	:return: indexes for data of the nearest neighbors """
	closest_indexes = [i for i in range(neighbors)]
	distances = [distance(data[i*2*same_trajectory_proximity_ban],state) for i in range(neighbors)]
	cmax = max(distances)
	# loop over individual states in data
	for i in range(len(data)-last_index_offset):
		too_big = False
		# loop over dimensions of the states
		for d in range(len(state)):
			if(abs(data[i][d]-state[d]) > cmax):
				too_big = True
				break
		if(too_big):
			continue
		# if it has a chance to be closer than the most distant of the current ones, check it properly and add it to the list of closest
		dist = distance(data[i],state)
		# same trajectory proximity test, if it is on the same trajectory compare it to the one already in the list and replace it if it is closer
		time_proximity = [abs(closest_indexes[j]-i) for j in range(neighbors)]
		proximity = min(time_proximity)
		if(proximity < same_trajectory_proximity_ban):
			index = time_proximity.index(proximity)
			if(distances[index] > dist):
				closest_indexes[index] = i
				distances[index] = dist
		# if it is not on the same trajectory, just a regular point, add it if its closer than the furthest one
		elif(dist < max(distances)):
			closest_indexes[distances.index(max(distances))] = i
			distances[distances.index(max(distances))] = dist
			cmax = max(distances)
	return closest_indexes


def nn_equiphase_points(data, pert_data, thr, T0_arg, neighbors, initial_state=None, sampling=10, warmup_periods=3, expansion='const', evolution_step=1):
	""" yields 'sampling' number of equiphase points on the limit cycle
	
	:param data: trajectory observations, data[0] are x(t), data[1] are y(t), data[2] are z(t) etc.
	:param pert_data: perturbation observations, pert_data[0] are p1(t), pert_data[1] are p2(t) etc.
	:param thr: threshold for signal x that determined zero phase
	:param T0_arg: the natural period measured in data steps, data[i][j] = data[i+n*T0_arg][j]
	:param neighbors: how many closest neighbours are considered
	:param initial_state: initial state of the system, [x,y,z,...] (dafault the first state from the data)
	:param sampling: phase sampling (default 10)
	:param warmup_periods: warmup periods to reach the limit cycle (default 3)
	:param expansion: expansion of f in \dot{x} = f(x) interpretation (default 'const')
	:param evolution_step: the evolution_step parameter used in nn_integration function for phase determination (default 1)
	:return: 'sampling' number of equiphasal points on the limit cycle as an array [[x0,y0,z0],[x1,y1,z1],...]
	"""
	# express the natural period in effective steps
	T0 = T0_arg/evolution_step
	# set the integration function
	int_function = set_integration_function(expansion)
	# initial conditions
	if(initial_state==None):
		state = [data[i][0] for i in range(len(data))]
	else:
		state = initial_state
	# integrate for 'warmup_periods' periods to get to the limit cycle
	for i in range(ceil(warmup_periods*T0)):
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# reach phase zero (threshold crossing)
	state_prev = state.copy()
	while((state_prev[0] < thr and state[0] > thr) == False):
		state_prev = state.copy()
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	# small correction due to overshooting
	state = [state_prev[i] + (thr-state_prev[0])/(state[0]-state_prev[0])*(state[i]-state_prev[i]) for i in range(len(state))]
	# move to the first phase to measure (half step)
	for i in range(int(round(0.5*T0/sampling))): 
		state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	equi_points = []
	for p in range(sampling):
		equi_points.append(state)
		for i in range(int(round((p+1.5)*T0/sampling)-round((p+0.5)*T0/sampling))): 
			state = int_function(data, pert_data, state, neighbors, evolution_step=evolution_step)
	return equi_points


def set_integration_function(expansion):
	""" assignes an integration function according to the 'expansion' parameter"""
	if(expansion=='linear'):
		return nn_one_step_integrate_linear
	elif(expansion=='const'):
		return nn_one_step_integrate_constant
	elif(expansion=='none'):
		return nn_one_step_no_input
	else:
		return nn_one_step_integrate_constant # if the argument is faulty set the expansion as constant by default



#########################################
#####   GENERAL HELPING FUNCTIONS   #####
#########################################


def sgn(x):
	"""a simple sign function"""
	if(x < 0):
		return -1
	return 1

def distance(point1, point2):
	"""L2 distance"""
	return sum((point1[i]-point2[i])**2 for i in range(len(point1)))**0.5

def distance1(point1, point2):
	"""L1 distance"""
	return sum(abs(point1[i]-point2[i]) for i in range(len(point1)))

def angle(x, y):
	"""angle of (x,y) with respect to origin (2D only)"""
	if(x == 0):
		if(y > 0):
			return pi/2
		return 3*pi/2
	elif(x < 0):
		return atan(y/x)+pi
	elif(y < 0):
		return atan(y/x)+2*pi
	return atan(y/x)

def rotate(direction, angl):
	"""rotates the direction by an angle (2D only)"""
	angle_original = angle(direction[0], direction[1])
	angle_total = angl+angle_original
	R = sqrt(direction[0]**2+direction[1]**2)
	return [R*cos(angle_total), R*sin(angle_total)]


