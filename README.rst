.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

**NEAREST NEIGHBOR PHASE RECONSTRUCTION** is a simple standalone code package for inferring the phase model form perturbed observations using nearest neighbor interpolation. 
All the functions for inference from observations are in file 'nn_phase_rec.py', while file 'eq_phase_rec.py' contains basic functions for obtaining the phase model from dynamical equations for comparison in tests. 

The basis for the inference is the local time evolution predictor, mapping the current state with the state at the next timestep: :math:`\textbf{x}(t) \rightarrow \textbf{x}(t+\Delta t)`. The local predictor is then simply nested to estimate longer time evolution with which the asymptotic phase is evaluated. 

**Required input:**
	observations of the state in time, :math:`\big[ \textbf{x}(t_1), \textbf{x}(t_2), \textbf{x}(t_3),...\big]`, where the state can be multidimensional :math:`\textbf{x} = (x_1,x_2,x_3,...x_n)`. 
**Optional input:**
	observations of perturbing forces in time, :math:`\big[ \textbf{p}(t_1), \textbf{p}(t_2), \textbf{p}(t_3),...\big]`, again they may be multidimensional :math:`\textbf{p} = (p_1,p_2,p_3,...p_m)` . Inclusion of perturbing observations generally improves the inference. 

**The main inference functions are:**
	- `nn_integrate()` - integrates an arbitrary state in time,
	- `nn_limit_cycle()` - estimates the limit cycle, 
	- `nn_period()` - estimates the natural period, 
	- `nn_phase()` - estimates the asymptotic phase, 
	- `nn_prc()` - estimates the phase response curve, 
	- `nn_iso()` - estimates an isochron (curve of constant phase).  

All functions are documented with docstrings, run :bash:`pydoc nn_phase_rec` in the same directory to see its compilation. 

**The main parameters are:**
	- `neighbors` - the number of neighbors considered, 

	- `expansion` - the dynamical expansions considered (3 options: `linear, constant, none`),
		assuming the dynamics can be described with an ODE:
		
		.. math::
			\dot{\textbf{x}} = \textbf{f}(\textbf{x}) + \sum_{i=1}^m \textbf{e}_i p_i(t)

		where :math:`\textbf{x} \in \mathbb{R}^n` is the state vector, :math:`\textbf{p} = [p_1,p_2,... p_m]` represents perturbations in all directions and :math:`\textbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^n` represents the inherent dynamics of the system. The problem is then approached with one of three expansions: 
			- approximating :math:`\textbf{f}` as a locally `linear` function: :math:`\dot{x}_j = a_{j0} + \sum_{i=1}^n a_{ji}x_i + \sum_{i=1}^m e_{ij}p_i(t)`,
			- approximating :math:`\textbf{f}` as a locally `constant` function: :math:`\dot{x}_j = a_{j0} + \sum_{i=1}^m e_{ij}p_i(t)`,
			- simply averaging evolutions of nearest neighbors (option `none`). This option does not require any knowledge of perturbing forces. 

**Simple example:**

Suppose we have data of a biological oscillatory process :python:`[x,y]` in time: :python:`data = [[x(t_1),x(t_2),x(t_3),...x(t_N)],[y(t_1),y(t_2),y(t_3),...y(t_N)]]` and a force :python:`p` perturbing it: :python:`pert = [p(t_1),p(t_2),p(t_3),...p(t_N)]`. If you have no perturbing force just replace it with an arbitrary array :python:`pert = [0 for i in range(len(data[0]))]` and set the :python:`expansion` parameter in all methods to :python:`'none'`. 

If we want to evaluate how an arbitrary state :python:`[x_0,y_0]` will evolve in time we run:

.. code:: python

	trajectory = nn_integrate(data, pert, number_of_steps, [x_0,y_0], neighbors, expansion='constant')

we have to define the :python:`number_of_steps` we want the trajectory to contain (time duration of steps the same as in :python:`data`), as well as the number of nearest :python:`neighbors` we want to consider. 

To evaluate the natural period of the oscillation we run:

.. code:: python

	T0 = nn_period(data, pert, threshold, neighbors, expansion='constant')

we further have to set a :python:`threshold` which determines which :python:`x` value counts as the beginning of the period. Normally the result should not depend on it as long as a common value of :python:`x` is chosen (mean value is a reasonable first choice: :python:`threshold = mean(data[0])`). 

To evaluate the phase response curve we then run:

.. code:: python

	prc = nn_prc(data, pert, threshold, T0, [x_dir,y_dir], neighbors, expansion='constant')

here we have to choose a direction :python:`[x_dir,y_dir]` with which the phase response curve will be evaluated (e.g. :python:`[1,0]` for the :python:`x` direction). 

**A complete example** is also found in file 'example.py'. Its data is automatically generated, from which the phase response curve is estimated and compared with the true one obtained directly from equations. 

