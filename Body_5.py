import sys
sys.path.insert(0, '../..')
import pyrosim # noqa
import math
import random
import numpy as np
from pyeasyga import pyeasyga

def evolve_gaits(weights):
	
	c = 0;
	dt = 1;

	sim = pyrosim.Simulator(play_paused=False, debug=True, use_textures=True,
                        	xyz=[3.0, 2.0, 2.0], hpr=[-150.0, -25.0, 0.0],#3 2 2 -150 -25 0   #-3 2 2 -30 -25 0
                        	eval_time=500, dt = 0.1)

	# DEFINING THE BODY PARAMETERS

	box = sim.send_box(x = 0.0,y = 0.0, z = 0.7, width = 1, length = 2.2, height = 0.1, mass = 10)

	thigh_cyl = [0]*6
	shin_cyl = [0]*6
	hip_hinge_joint = [0]*6
	knee_hinge_joint = [0]*6
	motor_neuron = [0]*6
	hidden_neuron = [0]*6

	x_pos_thigh = [0.75,0.75,0.75,-0.75,-0.75,-0.75]
	y_pos_thigh = [0,1,-1,0,1,-1]
	ax = [1,1,1,-1,-1,-1]


	x_pos_shin = [1,1,1,-1,-1,-1]
	y_pos_shin = [0,1,-1,0,1,-1]


	x_pos_hip_hinge = [0.5,0.5,0.5,-0.5,-0.5,-0.5]
	y_pos_hip_hinge = [0,1,-1,0,1,-1]

	x_pos_knee_hinge = [1,1,1,-1,-1,-1]
	y_pos_knee_hinge = [0,1,-1,0,1,-1]


	for i in range(0,6):
		thigh_cyl[i] = sim.send_cylinder(x = x_pos_thigh[i], y = y_pos_thigh[i], z = 0.7, r1 = ax[i], r2 = 0, r3 = 0, length = 0.5, radius = 0.05)
		shin_cyl[i] = sim.send_cylinder(x = x_pos_shin[i], y = y_pos_shin[i], z = 0.575, r1 = 0, r2 = 0, r3 = 1, length = 0.25, radius = 0.05)

	for i in range(0,6):
		hip_hinge_joint[i] = sim.send_hinge_joint(first_body_id = box,second_body_id = thigh_cyl[i], x = x_pos_hip_hinge[i], y = y_pos_hip_hinge[i], z = 0.7, n1 = 0, n2 = 1, n3 = 1, lo=-math.pi/2, 							  hi = math.pi/2)
	
		knee_hinge_joint[i] = sim.send_hinge_joint(first_body_id = thigh_cyl[i],second_body_id = shin_cyl[i], x = x_pos_knee_hinge[i], y = y_pos_knee_hinge[i], z = 0.7, n1 = 0, n2=1,n3=0,lo = 0,  							           hi = 0)



	## INCLUDE SENSORS

	# MAIN BODY SENSORS
	pos_box_sensor = sim.send_position_sensor(box)
	touch_box_sensor = sim.send_touch_sensor(box)


	# Sending function neuron
	input_neuron = sim.send_function_neuron(math.sin)

	# Sending a bias neuron
	#bias_neuron  = sim.send_bias_neuron()

	# Sending hidden neurons
	for i in range(0,6):
		hidden_neuron[i] = sim.send_hidden_neuron()


	# Sending motor neurons to each hip joint
	for i in range(0,6):
		motor_neuron[i] = sim.send_motor_neuron(hip_hinge_joint[i])





	# Generating Synapses

	for i in range(0,6):
		#sim.send_synapse(bias_neuron,hidden_neuron[i],weights[0,i])
		sim.send_synapse(input_neuron,hidden_neuron[i],weights[0,i])
	

	for i in range(0,6):
		for j in range(0,6):
			sim.send_synapse(hidden_neuron[i],motor_neuron[j],weights[i+1,j])


	sim.create_collision_matrix('all')


	sim.start()
	sim.wait_to_finish()

	pos_box_y_results = sim.get_sensor_data(pos_box_sensor, svi=1)
	pos_box_x_results = abs(sim.get_sensor_data(pos_box_sensor, svi=0))
	x_values = np.sum(pos_box_x_results)
	touch_sensor_results = sim.get_sensor_data(touch_box_sensor, svi=0)
	touches = np.sum(touch_sensor_results)
    

	return(pos_box_y_results[-1],x_values,touches)






