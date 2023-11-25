#!/usr/bin/env python3

# This is the code you need to do the Bayes' update for the robot in front of the door problem

import numpy as np

from world_ground_truth import WorldGroundTruth
from robot_sensors import RobotSensors
from robot_ground_truth import RobotGroundTruth

class BayesFilter:
    def __init__(self):

        self.probabilities = []
        self.directions = ["left", "stay_put", "right"]
        self.reset_probabilities()

    def reset_probabilities(self, n_bins: int = 10):
        """ Initialize discrete probability resolution with uniform distribution
        @param n_bins - the number of bins to divide the unit interval (0,1) up into """
        prob = (1) / n_bins
        self.probabilities = prob*np.ones((n_bins))

    def update_belief_sensor_reading(self, world_ground_truth, robot_sensor, sensor_reading):
        """ Update your probabilities based on the sensor reading being true (door) or false (no door)
        Slide: Lec 1.2, Bayes rule sensor
         https://docs.google.com/presentation/d/1p_QI9BFHgEiwoWdMApBvyr4glhdVgX3iRcA7G3ST2jo/edit#slide=id.p2

        @param world_ground_truth - has where the doors actually are
        @param robot_sensor - has the robot sensor probabilities
        @param sensor_reading - the actual sensor reading - either True or False
        """
        # bayes assignment
        # Note: it's easier to implement this by making a new probabilities array, filling it up, then assigning the
        #  new values to self.probabilities during the normalization step (dividing by nu)
        # Don't forget to normalize - that's the divide by nu part. This is because all of the denominators in the
        #  update have the same value (which, conveniently, is the sum of the numerators)
        
        # TODO
        w = 1 / len(self.probabilities)
        init = 0        
        new_probs = np.zeros(len(self.probabilities))
        bin_locs = []
        for door_loc in world_ground_truth.doors:
            #make a list of dorr locations
            loc = np.floor((door_loc - init)/w) 
            bin_locs.append(int(loc)) 
        #update probabilities posterior dist
        for nbin, p in enumerate(self.probabilities): 
            if sensor_reading and nbin in bin_locs: 
                new_probs[nbin] = p * (robot_sensor.sensor_prob_dict ["door"]["see"])
            elif sensor_reading and not nbin in bin_locs: 
                new_probs[nbin] = p * (robot_sensor.sensor_prob_dict ["no_door"]["see"])
            elif not sensor_reading and nbin in bin_locs:
                new_probs[nbin] = p * (robot_sensor.sensor_prob_dict ["door"]["nosee"])
            else:
                new_probs[nbin] = p * (robot_sensor.sensor_prob_dict ["no_door"]["nosee"])
    
        self.probabilities = new_probs / np.sum(new_probs) #normalize

    def update_belief_move_left(self, robot_ground_truth):
        """ Update the probabilities assuming a move left.
        Slide: Lec 2.1 Bayes rule actions
        https://docs.google.com/presentation/d/1Vjiy4kqmuYWq811huXZNAU2FGSXNfI3me54oW1ko6hU/edit#slide=id.g16f70ac0d7c_0_9

        Note that transitions p(x|x',u) in the slides is stored in robot_ground_truth.move_probabilities["move_left"], for all the
        middle bins - you'll have to special-case the transitions at the first and last bins
        @param robot_ground_truth - robot location, has the probabilities for actually moving left if move_left called"""

        # bayes assignment
        # Note: it's easier to implement this by making a new probabilities array (filled with zeros) then assigning the
        #  new values to self.probabilities during the normalization step
        # Don't forget to normalize - but if you've done this correctly then the sum should be very, very close to
        #  one already - any error is just numerical
        nbinss = len(self.probabilities)
        new_probs = np.zeros((nbinss))
        for idx, p in enumerate(self.probabilities):
            #handling left and right wall
            left_idx = max(0, idx-1) 
            right_idx = min(nbinss-1, idx+1)
            #updating probabilities posterior
            new_probs[right_idx] += robot_ground_truth.move_probabilities["move_left"]["right"] * p
            new_probs[left_idx] += robot_ground_truth.move_probabilities["move_left"]["left"] * p
            new_probs[idx] += robot_ground_truth.move_probabilities["move_left"]["stay_put"] * p

        self.probabilities = new_probs / np.sum(new_probs) #normalize

    def update_belief_move_right(self, robot_ground_truth):
        """ Update the probabilities assuming a move right.
        Sames as above - but this time, transitions are stored in robot_ground_truth.move_probabilities["move_right"]
        @param robot_ground_truth - robot location, has the probabilities for actually moving left if move_left called"""
        #same as above just for right side
        nbinss = len(self.probabilities)
        new_probs = np.zeros((nbinss)) 
        for idx, p in enumerate(self.probabilities):
            left_idx = max(0, idx-1) 
            right_idx = min(nbinss-1, idx+1) 
            new_probs[right_idx] += robot_ground_truth.move_probabilities["move_right"]["right"] * p
            new_probs[left_idx] += robot_ground_truth.move_probabilities["move_right"]["left"] * p
            new_probs[idx] += robot_ground_truth.move_probabilities["move_right"]["stay_put"] * p
            
        self.probabilities = new_probs / np.sum(new_probs) #normalize
        

    def one_full_update(self, world_ground_truth, robot_ground_truth, robot_sensor, u: str, z: bool):
        """This is the full update loop that takes in one action, followed by a sensor reading
        Lec 2_2 State estimation, Bayes filter algorithm section
        Slides: https://docs.google.com/presentation/d/1V_l8GNlGgkvzVMheff_tM7tJlJ4A1y-363QB5hQpHVQ/edit#slide=id.g14a2b1b8e73_0_396
        Assumes the robot has been moved by the action u, then a sensor reading was taken (see test function below)
        @
        @param world_ground_truth - has where the doors actually are
        @param robot_sensor - has the robot sensor probabilities
        @param robot_ground_truth - robot location, has the probabilities for actually moving left if move_left called
        @param u will be one of "move_left" or "move_right" (string)
        @param z will be one of True or False (door y/n)
        """
        if u == "move_left":
            self.update_belief_move_left(robot_ground_truth)
        if u == "move_right":
            self.update_belief_move_right(robot_ground_truth)
        self.update_belief_sensor_reading(world_ground_truth, robot_sensor, z)


def check_uniform(bf):
    """ At the start, should be uniform probability
    @param bf - the bayes filter at initialization"""

    if not np.isclose(np.sum(bf.probabilities), 1.0):
        raise ValueError(f"Check uniform, expected sum to one, got {np.sum(bf.probabilities)}")

    n_per = 1.0 / len(bf.probabilities)
    for p in bf.probabilities:
        if not np.isclose(p, n_per):
            raise ValueError(f"Check uniform, expected {n_per} in all bins, got {p}")
    return True


def check_door_probs(bayes_filter, world_ground_truth, robot_sensor_probs, sensor_readings, b_print=True):
    """ If only doing door updates, there are a couple things that have to hold true
     1) All in front of door (and not in front of door) probabilities should be the same
     2) those probabilities are the same as rolling dice n times
     @param bayes_filter - the bayes filter after the sensor readings
     @param world_ground_truth - the world ground truth/door readings
     @param robot_sensor_probs - the robot sensor probabilities
     @param sensor_readings - list of True/False sensor reading return values
     @param b_print - do print statements, yes/no"""

    if b_print:
        print(f"Checking sequence {sensor_readings}")

    n_doors = len(world_ground_truth.doors)
    n_bins = len(bayes_filter.probabilities)
    div_bins = 1.0 / n_bins

    # First, calculate what the probability of being in front of a door is, given the sensor values (assuming in front
    #   of door)
    prob_in_front_of_door = div_bins
    prob_not_in_front_of_door = div_bins
    for reading in sensor_readings:
        if reading:
            prob_in_front_of_door = prob_in_front_of_door * robot_sensor_probs[0]
            prob_not_in_front_of_door = prob_not_in_front_of_door * robot_sensor_probs[1]
        else:
            prob_in_front_of_door = prob_in_front_of_door * (1.0 - robot_sensor_probs[0])
            prob_not_in_front_of_door = prob_not_in_front_of_door * (1.0 - robot_sensor_probs[1])

    # Normalize by number of doors
    sum_probs = n_doors * prob_in_front_of_door + (n_bins - n_doors) * prob_not_in_front_of_door
    prob_in_front_of_door /= sum_probs
    prob_not_in_front_of_door /= sum_probs
    for i_bin, prob in enumerate(bayes_filter.probabilities):
        loc = (i_bin+0.5) * div_bins
        if world_ground_truth.is_location_in_front_of_door(loc):
            if not np.isclose(prob, prob_in_front_of_door):
                raise ValueError(f"Check door probabilities: new probability should be {prob_in_front_of_door}, was{prob}")
        else:
            if not np.isclose(prob, prob_not_in_front_of_door):
                raise ValueError(f"Check door probabilities: new probability should be {prob_not_in_front_of_door}, was {prob}")

    if b_print:
        print("Passed\n")
    return True


def test_bayes_filter_sensor_update(b_print=True):
    """ Do a sensor update with known values and check that the answer is correct
    How this works: If there is no motion (just sensor readings) than the values for all of the locations in front
     of the doors should be the same (also true for all of the locations NOT in front of the doors)
     Those probabilities are just the product of the Bayes' update rule
    @param b_print - do print statements, yes/no"""
    world_ground_truth = WorldGroundTruth()
    robot_sensor = RobotSensors()
    bayes_filter = BayesFilter()

    n_doors = 2
    n_bins = 20
    probs = (0.6, 0.1)

    if b_print:
        print("Checking Bayes filter sensor update")
    np.random.seed(2)

    # Initialize with values that are NOT the default ones
    world_ground_truth.random_door_placement(n_doors, n_bins)
    robot_sensor.set_door_sensor_probabilites(probs[0], probs[1])

    # The sequences to try. You can add more if you'd like. The first two check the True and False cases
    seqs = [[True], [False], [True, True, False]]
    for seq in seqs:
        # Double check that you're starting off with uniform probabilities
        bayes_filter.reset_probabilities(n_bins)
        check_uniform(bayes_filter)

        # Call the update function with the given sensor readings
        for s in seq:
            bayes_filter.update_belief_sensor_reading(world_ground_truth, robot_sensor, s)

        # The actual check function
        check_door_probs(bayes_filter, world_ground_truth, probs, seq, b_print)

    if b_print:
        print("Passed all sequences\n")
    return True


def test_move_one_direction(b_print=True):
    """ Move all the way to the left (or the right) a LOT, so should pile up probability in the left (or right) bin
    Use the default probabilities
     @param b_print - do print statements, yes/no"""
    bayes_filter = BayesFilter()
    robot_ground_truth = RobotGroundTruth()

    n_bins = 15
    step_size = 1.0 / n_bins
    n_moves = n_bins * 10

    if b_print:
        print("Testing move in one direction")
    np.random.seed(20)

    # Try the left, then the right move
    # Python note: you can treat class methods/functions just like other variables - this creates a tuple with the
    #   two methods (move left, move right) so we can call them in the for loop (dir_move)
    dirs_move = (robot_ground_truth.move_left, robot_ground_truth.move_right)
    dirs_update = (bayes_filter.update_belief_move_left, bayes_filter.update_belief_move_right)
    for dir_move, dir_update, bin_id in zip(dirs_move, dirs_update, (0, n_bins-1)):
        if b_print:
            print(f"Test move {dir_move.__name__}")
        # Reset both the bayes filter probabilities and the robot ground truth location
        bayes_filter.reset_probabilities(n_bins)
        robot_ground_truth.reset_location()

        for _ in range(0, n_moves):
            # Move the robot
            dir_move(step_size)
            # Update the bayes filter - needs the probabilities in robot_ground_truth (NOT the robot's actual location)
            dir_update(robot_ground_truth)

        if not bayes_filter.probabilities[bin_id] > 0.9:
            raise ValueError(f"Expected all of the probability to be in the {bin_id} bin, was {bayes_filter.probabilities[bin_id]}")
        if b_print:
            print("Passed")

    return True




def test_move_update(b_print=True):
    """ Test the move update. This test is done by comparing your probability values to some pre-calculated/saved values
    @param b_print - do print statements, yes/no"""

    bayes_filter = BayesFilter()
    world_ground_truth = WorldGroundTruth()
    robot_ground_truth = RobotGroundTruth()
    robot_sensor = RobotSensors()

    # Read in some move sequences and compare your result to the correct answer
    import json
    with open("Data/check_bayes_filter.json", "r") as f:
        answers = json.load(f)

    n_doors = answers["n_doors"]
    n_bins = answers["n_bins"]
    step_size = 1.0 / n_bins
    world_ground_truth.doors = answers["world"]

    if b_print:
        print("Testing move update")

    # This SHOULD insure that you get the same answer as the solutions, provided you're only calling uniform within
    #  robot_ground_truth.move and robot_sensor.query door*
    seed = 3
    np.random.seed(seed)

    # Try different probability values
    for answer in answers["answers"]:
        # This SHOULD insure that you get the same answer as the solutions, provided you're only calling uniform within
        #  robot_ground_truth.move and robot_sensor.query_door*
        np.random.seed(seed)
        seed = seed + 1

        seq = answer["seq"]
        zs = answer["Sensors"]

        # Reset to uniform
        bayes_filter.reset_probabilities(n_bins)
        robot_ground_truth.reset_location()
        for z_check, s in zip(zs, seq):
            if s == "left":
                robot_ground_truth.move_left(step_size)
            elif s == "right":
                robot_ground_truth.move_right(step_size)
            else:
                raise ValueError(f"Expected left or right, got {s}")

            z = robot_sensor.query_door(robot_ground_truth, world_ground_truth)
            if b_print:
                print(f"Loc {robot_ground_truth.robot_loc}, Doors {world_ground_truth.doors}, Sensor {z}")
            if z is not z_check:
                print(f"Warning: Sensor reading is different than check {z} versus {z_check}")

            bayes_filter.one_full_update(world_ground_truth, robot_ground_truth, robot_sensor, "move_" + s, z)

        check_seed = np.random.uniform()
        if not np.isclose(check_seed, answer["check_seed"]):
            print("Warning: random number generator is off, may report incorrect result")

        if not np.any(np.isclose(answer["result"], bayes_filter.probabilities, atol=0.01)):
            ValueError(f"Probabilities are different \n{answer['result']} \n{bayes_filter.probabilities}")

    if b_print:
        print("Passed")
    return True


if __name__ == '__main__':
    b_print_test = True

    # Syntax checks
    n_doors_syntax = 2
    n_bins_syntax = 10
    world_ground_truth_syntax = WorldGroundTruth()
    world_ground_truth_syntax.random_door_placement(n_doors_syntax, n_bins_syntax)
    robot_sensor_syntax = RobotSensors()
    bayes_filter_syntax = BayesFilter()
    robot_ground_truth_syntax = RobotGroundTruth()

    # Syntax check 1, reset probabilities
    bayes_filter_syntax.reset_probabilities(n_bins_syntax)

    # Syntax check 2, update sensor
    bayes_filter_syntax.update_belief_sensor_reading(world_ground_truth_syntax, robot_sensor_syntax, True)

    # Syntax check 3, move
    bayes_filter_syntax.update_belief_move_left(robot_ground_truth_syntax)
    bayes_filter_syntax.update_belief_move_right(robot_ground_truth_syntax)

    # Syntax check 4, full update
    bayes_filter_syntax.one_full_update(world_ground_truth_syntax, robot_ground_truth_syntax, robot_sensor_syntax, "move_left", True)

    # The tests
    test_bayes_filter_sensor_update(b_print_test)
    test_move_one_direction(b_print_test)


    test_move_update(b_print_test)

    print("Done")