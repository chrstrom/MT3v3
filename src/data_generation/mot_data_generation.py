import numpy as np
import itertools


def get_measurement_from_object(obj):
    r = np.linalg.norm(obj.pos)
    theta = np.arctan2(obj.pos[1], obj.pos[0])
    return np.array([r, theta])


class FieldOfView:
    def __init__(self, min_range, max_range, min_theta, max_theta):
        self.min_range = min_range
        self.max_range = max_range
        self.min_theta = min_theta
        self.max_theta = max_theta

    def __contains__(self, measurement):
        if not (self.min_range <= measurement[0] <= self.max_range):
            return False
        elif not (self.min_theta <= measurement[1] <= self.max_theta):
            return False
        else:
            return True

    def area(self):
        range_length = self.max_range - self.min_range
        theta_length = self.max_theta - self.min_theta
        return range_length * theta_length


class Object:

    def __init__(self, pos, vel, t, delta_t, sigma, id):
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t
        self.sigma = sigma
        self.state_history = np.array([np.concatenate([pos, vel, np.array([t])])])
        self.process_noise_matrix = sigma*np.array([[delta_t ** 3 / 3, delta_t ** 2 / 2], [delta_t ** 2 / 2, delta_t]])
        self.id = id

    def update(self, t, rng):
        """
        Updates this object's state using a discretized constant velocity model.
        """

        # Update position and velocity of the object in each dimension separately
        assert len(self.pos) == len(self.vel)
        process_noise = rng.multivariate_normal([0, 0], self.process_noise_matrix, size=len(self.pos))
        self.pos += self.delta_t * self.vel + process_noise[:,0]
        self.vel += process_noise[:,1]

        # Add current state to previous states
        current_state = np.concatenate([self.pos.copy(), self.vel.copy(), np.array([t])])
        self.state_history = np.vstack((self.state_history, current_state))

    def __repr__(self):
        return 'id: {}, pos: {}, vel: {}'.format(self.id, self.pos, self.vel)


class MotDataGenerator:
    def __init__(self, args, rng):
        
        gen_args = args.data_generation
        
        
        if not (isinstance(gen_args.measurement_noise_stds, list) and
                len(gen_args.measurement_noise_stds)==2):
            raise ValueError(f"Specified measurement noise should be a list with two elements, got "
                             f"'{gen_args.measurement_noise_stds}' instead")

        self.delta_t = gen_args.dt

        self.start_pos_params = [gen_args.birth_process.mean_pos, gen_args.birth_process.cov_pos]
        self.start_vel_params = [gen_args.birth_process.mean_vel, gen_args.birth_process.cov_vel]
        
        self.lambda_birth_in = rng.uniform(gen_args.lambda_b_in[0], gen_args.lambda_b_in[1])
        self.lambda_birth_enter = rng.uniform(gen_args.lambda_b_enter[0], gen_args.lambda_b_enter[1])
        self.prob_survival = rng.uniform(gen_args.p_survival[0], gen_args.p_survival[1])
        self.process_noise_variance = rng.uniform(gen_args.process_noise_variance[0], gen_args.process_noise_variance[1])
        self.prob_measure = rng.uniform(gen_args.p_meas[0], gen_args.p_meas[1])
        
        
        self.measurement_noise_stds = gen_args.measurement_noise_stds
        
        self.x_size = 2 # Number of elements in measurements
        
        self.n_average_false_measurements = rng.uniform(gen_args.lambda_clutter[0], gen_args.lambda_clutter[1])
        self.n_average_starting_objects = gen_args.n_avg_starting_objects
        
        
        field_of_view_min_theta = gen_args.field_of_view.min_theta if gen_args.field_of_view.min_theta is not None else -np.pi
        field_of_view_max_theta = gen_args.field_of_view.max_theta if gen_args.field_of_view.max_theta is not None else np.pi
        
        self.field_of_view = FieldOfView(gen_args.field_of_view.min_range,
                                         gen_args.field_of_view.max_range,
                                         field_of_view_min_theta,
                                         field_of_view_max_theta)
        
        self.max_objects = gen_args.max_objects
        if f'get_{gen_args.prediction_target}_from_state' in globals():
            self.prediction_target = gen_args.prediction_target
        else:
            raise NotImplementedError(f'The chosen function for mapping state to ground-truth was no implemented: {gen_args.prediction_target}')
        
        self.rng = rng
        self.debug = False


        for s_y in gen_args.measurement_noise_stds:
            assert s_y is not None, f"Measurement noise cannot be None. Got {gen_args.measurement_noise_stds}."

        assert self.n_average_starting_objects != 0, 'Datagen does not currently work with n_avg_starting_objects equal to zero.'

        self.t = None
        self.objects = None
        self.trajectories = None
        self.measurements = None
        self.true_measurements = None
        self.false_measurements = None
        self.object_events = None  # (time, id_birth, id_death) id -1 if no event
        self.unique_ids = None
        self.unique_id_counter = None
        self.reset()

    def reset(self):
        self.t = 0
        self.objects = []
        self.object_events = []
        self.trajectories = {}
        self.measurements = np.array([])
        self.true_measurements = np.array([])
        self.false_measurements = np.array([])
        self.unique_ids = np.array([], dtype="int64")
        self.unique_id_counter = itertools.count()


        # Add initial set of objects (re-sample until we get a nonzero value)
        n_starting_objects = 0
        while n_starting_objects == 0:
            n_starting_objects = self.rng.poisson(self.n_average_starting_objects)
        self.add_objects(n_starting_objects)

        # Measure the initial set of objects
        self.generate_measurements()

        if self.debug:
            print(n_starting_objects, 'starting objects')

    def create_normal_object(self, pos, vel):
        return Object(pos=pos,
                      vel=vel,
                      t=self.t,
                      delta_t=self.delta_t,
                      sigma=self.process_noise_variance,
                      id=next(self.unique_id_counter))
        
    def create_entry_object(self):
        r = self.field_of_view.max_range - self.rng.uniform(low=0.1, high=1.0)
        theta = self.rng.uniform(low=self.field_of_view.min_theta, high=self.field_of_view.max_theta)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        pos = np.array((x, y))
        velocity_vector_angle_towards_sensor = np.arctan2(-y, -x)
        alpha = np.pi / 4 # max deviation from center line of cone
        beta = self.rng.uniform(low=-alpha, high=alpha)
        direction = velocity_vector_angle_towards_sensor + beta
        
        velocity_amplitude = np.abs(self.rng.normal(0, np.sqrt(15)))
        
        vx = velocity_amplitude * np.cos(direction)
        vy = velocity_amplitude * np.sin(direction)
        vel = np.array((vx, vy))
        
        return Object(pos=pos,
                      vel=vel,
                      t=self.t,
                      delta_t=self.delta_t,
                      sigma=self.process_noise_variance,
                      id=next(self.unique_id_counter))

    def create_n_objects(self, n):
        """
        Create `n` objects according to Gaussian birth model. Objects outside measurement FOV are discarded.
        """
        positions = self.rng.multivariate_normal(self.start_pos_params[0], self.start_pos_params[1], size=(n,))
        velocities = self.rng.multivariate_normal(self.start_vel_params[0], self.start_vel_params[1], size=(n,))
        objects = []
        for pos, vel in zip(positions, velocities):
            obj = self.create_normal_object(pos, vel)
                
            if get_measurement_from_object(obj) in self.field_of_view:
                objects.append(obj)
        return objects
    
    def create_n_entry_objects(self, n):
        """
        Create `n` objects according to Gaussian birth model. Objects outside measurement FOV are discarded.
        """
        positions = self.rng.multivariate_normal(self.start_pos_params[0], self.start_pos_params[1], size=(n,))
        velocities = self.rng.multivariate_normal(self.start_vel_params[0], self.start_vel_params[1], size=(n,))
        objects = []
        for pos, vel in zip(positions, velocities):
            obj = self.create_entry_object()
                
            if get_measurement_from_object(obj) in self.field_of_view:
                objects.append(obj)
        return objects

    def add_normal_objects(self, n):
        n = min(n, self.max_objects - len(self.objects))
        if n > 0:
            new_objects = self.create_n_objects(n)
            for obj in new_objects:
                self.object_events.append(np.array((np.round(self.t, 3), obj.id, -1)))
            self.objects += new_objects

    def add_entry_object(self, n):
        n = min(n, self.max_objects - len(self.objects))
        if n > 0:
            new_entering_objects = self.create_n_entry_objects(n)
            for obj in new_entering_objects:
                self.object_events.append(np.array((np.round(self.t, 3), obj.id, -1)))
            self.objects += new_entering_objects

    def add_objects(self, n=None):
        if n is not None:
            self.add_normal_objects(n)
            return
        
        n = self.rng.poisson(self.lambda_birth_in)
        self.add_normal_objects(n)
        
        n = self.rng.poisson(self.lambda_birth_enter)
        self.add_entry_object(n)

    def remove_far_away_objects(self):

        if len(self.objects) == 0:
            return

        # Check which objects left the FOV
        meas_coordinates_of_objects = np.array(
            [get_measurement_from_object(obj) for obj in self.objects]
        )
        deaths = [
            meas_obj not in self.field_of_view
            for meas_obj in meas_coordinates_of_objects
        ]

        # Save state history of objects that will be removed in self.trajectories
        for obj, death in zip(self.objects, deaths):
            if death:
                self.trajectories[obj.id] = obj.state_history[:-1]
                self.object_events.append(np.array((np.round(self.t, 3), -1, obj.id)))

    def remove_objects(self, p):
        """
        Removes each of the objects with probability `p`.
        """

        # Compute which objects are removed in this time-step
        deaths = self.rng.binomial(n=1, p=p, size=len(self.objects))

        n_deaths = sum(deaths)
        if self.debug and (n_deaths > 0):
            print(n_deaths, "objects were removed")

        # Save the trajectories of the removed objects
        for obj, death in zip(self.objects, deaths):
            if death:
                self.trajectories[obj.id] = obj.state_history
                self.object_events.append(np.array((np.round(self.t, 3), -1, obj.id)))

        # Remove them from the object list
        self.objects = [o for o, d in zip(self.objects, deaths) if not d]

    # def get_prob_death(self, obj):
    #     return 1 - self.prob_survival

    def generate_measurements(self):
        """
        Generates all measurements (true and false) for the current time-step.
        """

        # Decide which of the objects will be measured
        is_measured = self.rng.binomial(n=1, p=self.prob_measure, size=len(self.objects))
        measured_objects = [obj for obj, is_measured in zip(self.objects, is_measured) if is_measured]

        # Generate the true measurements' noise
        true_measurements = []
        true_measurements_with_id = []

        measurement_noises = self.rng.normal(0, self.measurement_noise_stds, size=(len(measured_objects), self.x_size))

        # Generate true measurements (making sure they're inside the FOV)
        for i, obj in enumerate(measured_objects):
            m = get_measurement_from_object(obj)
            measurement_with_time = np.append(m + measurement_noises[i, :], self.t)
            if measurement_with_time[:-1] in self.field_of_view:
                true_measurements.append(measurement_with_time)
                true_measurements_with_id.append(
                    np.append(measurement_with_time, obj.id)
                )

        true_measurements = np.array(true_measurements)
        true_measurements_with_id = np.array(true_measurements_with_id)

        unique_obj_ids_true = [obj.id for obj in measured_objects]

        # Generate false measurements (uniformly distributed over measurement FOV)
        n_false_measurements = self.rng.poisson(self.n_average_false_measurements)
        false_measurements = \
            self.rng.uniform(low=[0, self.field_of_view.min_theta],
                             high=[1, self.field_of_view.max_theta],
                             size=(n_false_measurements, self.x_size))
            
        def uniform_radius_distr(x):
            return self.field_of_view.min_range + (self.field_of_view.max_range - self.field_of_view.min_range) * np.sqrt(x)
            
        false_measurements[:, 0] = uniform_radius_distr(false_measurements[:, 0])

        # Add time to false measurements
        times = np.repeat([[self.t]], n_false_measurements, axis=0)
        false_measurements = np.concatenate([false_measurements, times], 1)

        # Also save from which object each measurement came from (for contrastive learning later); -1 is for false meas.
        unique_obj_ids_false = [-1]*len(false_measurements)
        unique_obj_ids = np.array(unique_obj_ids_true + unique_obj_ids_false)

        # Concatenate true and false measurements in a single array
        if true_measurements.shape[0] and false_measurements.shape[0]:
            new_measurements = np.vstack([true_measurements, false_measurements])
        elif true_measurements.shape[0]:
            new_measurements = true_measurements
        elif false_measurements.shape[0]:
            new_measurements = false_measurements
        else:
            return

        # Shuffle all generated measurements and corresponding unique ids in unison
        random_idxs = self.rng.permutation(len(new_measurements))
        new_measurements = new_measurements[random_idxs]
        unique_obj_ids = unique_obj_ids[random_idxs]
        
        # Save measurements and unique ids
        self.measurements = (
            np.vstack([self.measurements, new_measurements])
            if self.measurements.shape[0]
            else new_measurements
        )
        if not true_measurements.size == 0:
            self.true_measurements = (
                np.vstack([self.true_measurements, true_measurements_with_id])
                if self.true_measurements.shape[0]
                else true_measurements_with_id
            )
        if not false_measurements.size == 0:
            self.false_measurements = (
                np.vstack([self.false_measurements, false_measurements])
                if self.false_measurements.shape[0]
                else false_measurements
            )
        self.unique_ids = np.hstack([self.unique_ids, unique_obj_ids])

    def step(self, add_new_objects=True):
        """
        Performs one step of the simulation.
        """
        self.t += self.delta_t

        for obj in self.objects:
            obj.update(self.t, self.rng)

        self.remove_far_away_objects()

        if add_new_objects:
            self.add_objects()

        self.remove_objects(1 - self.prob_survival)
        
        self.generate_measurements()
        

    def finish(self):
        """
        Should be called after the last call to `self.step()`. Removes the remaining objects, consequently adding the
        remaining parts of their trajectories to `self.trajectories`.
        """
        self.remove_objects(1.0)

        label_data = []
        unique_label_ids = []

        # -1 is applied because we count t=0 as one time-step
        last_timestep = round(self.t / self.delta_t)
        for traj_id in self.trajectories:
            traj = self.trajectories[traj_id]
            last_state = traj[-1]
            if round(last_state[4] / self.delta_t) == last_timestep:  # last state of trajectory, time
                pos = globals()[f'get_{self.prediction_target}_from_state'](last_state)
                label_data.append(pos)
                unique_label_ids.append(traj_id)
                
        training_data = np.array(self.measurements.copy())
        unique_measurements_ids = self.unique_ids.copy()
        true_measurements = np.array(self.true_measurements.copy())
        false_measurements = np.array(self.false_measurements.copy())

        return (
            training_data,
            label_data,
            unique_measurements_ids,
            unique_label_ids,
            true_measurements,
            false_measurements,
            self.object_events,
        )



def get_position_from_state(state):
    return state[:2].copy()


def get_position_and_shape_from_state(state):
    return np.concatenate((state[:2].copy(), state[5:9].copy()))


def get_position_and_velocity_from_state(state):
    return state[:4].copy()