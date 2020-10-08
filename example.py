from nn_phase_rec import *
from eq_phase_rec import *
from matplotlib import pyplot # for a default plot

# algorithm parameters
closest_howmany = 10
expansion = 'linear' 

# data parameters
sim_time = 100
DT = 0.005
N = int(round(sim_time/DT))
SAMPLING = 3
thr = 0.0

# the system
def dx(state,ps):
	return state[1] - sin(state[1])*state[0]/2 + ps[0]
def dy(state,ps):
	return -state[0] + cos(state[0])*state[1]/2 + ps[1]
ders = [dx,dy]

# generate perturbed signal
data, pert = generate_signal(ders, N, SAMPLING, number_of_variables=2, number_of_perturbations=2, tau=1, eps=0.2, dt=DT)

# estimate the phase response curve 
T0 = nn_period(data, pert, thr, closest_howmany, expansion=expansion)
prc_nn = nn_prc(data, pert, thr, T0, [1,0], closest_howmany, expansion=expansion)

# true phase response curve for comparison
prc_true = oscillator_PRC(ders, [1,0])

# plot
pyplot.plot(prc_true[0], prc_true[1], c='b')
pyplot.scatter([prc_nn[i][0] for i in range(len(prc_nn))], [prc_nn[i][1] for i in range(len(prc_nn))], c='r')
pyplot.xticks([0,pi/2,pi,3*pi/2,2*pi],[0,r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
pyplot.grid(linestyle=':')
pyplot.xlabel(r"$\varphi$")
pyplot.ylabel(r"$Z(\varphi)$")
pyplot.show()
