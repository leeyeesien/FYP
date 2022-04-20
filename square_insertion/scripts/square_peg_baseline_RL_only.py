import franka_cartesian_controller.motion_generator as motion_generator
import franka_cartesian_controller.transform_util as T
from peg_insertion_functions import InsertionMotions, Utilities

import numpy as np
from datetime import datetime
import time as mytime

from eval_real_square import evaluate_experiment

if __name__ == "__main__":
    robot = motion_generator.MotionGenerator(verbose="debug")
    insertion_motions = InsertionMotions()
    utilities = Utilities()

    file = "/../logs_ins_exe/log_msgs_{}.txt".format(datetime.now())
    i = 0

    starting_eef_pos = robot.eef_pos
    starting_eef_quat = robot.eef_quat
    utilities.print_log("\nend-effector starting translation wrt fixed base frame: \n{}".format(starting_eef_pos))
    utilities.print_log("\nend-effector starting rot_matrix wrt fixed base frame: \n{}".format(T.quat2rotation(starting_eef_quat)))

    # Go to goal pose
    robot.move_to_pose_request(
    target_pos=[starting_eef_pos[0], starting_eef_pos[1], 0.325],
    target_quat=starting_eef_quat,
    linear_speed=0.1,
    rotation_speed=0.4,
    max_duration=5
    )
    T_0A_pos_ref = robot.eef_pos
    T_0A_quat_ref = robot.eef_quat
    utilities.print_log("\nend-effector T_0A(ref:T_0A)_i_{} translation: \n{}".format(i, T_0A_pos_ref))
    utilities.print_log("\nend-effector T_0A(ref:T_0A)_i_{} rot_matrix: \n{}".format(i, T.quat2rotation(T_0A_quat_ref)))

    success_count = 0
    model_est_error = np.zeros(6)
    controller_error = np.zeros(6)
    insertion_time_list = [0] * 50

    for i in range(50):
        if i >= 1:
            robot.move_to_pose_request(
                target_pos=T_0A_pos_ref,
                target_quat=T_0A_quat_ref,
                linear_speed=0.1,
                rotation_speed=0.4,
                max_duration=5
            )

        utilities.print_log("\nAttempt {}\n".format(i+1))

        # Move to a random pose within the cylinder
        T_0B_pos, T_0B_quat = insertion_motions.move_to_random_pose_wo_image(T_0A_pos_ref, T_0A_quat_ref, "B(test:T_0B)_i_{}".format(i))

        mytime.sleep(1)

        contact_success, contact_time = robot.move_to_contact_request(
            direction=[0, 0, -1, 0, 0, 0],
            speed=0.01,
            force_thresh=10,
            max_duration=20,
        )

        utilities.print_log("Move to contact request success and time: \n")
        utilities.print_log(contact_success)
        utilities.print_log(contact_time)

        if contact_success is True:
            contact_eef_pos = robot.eef_pos
            contact_eef_quat = robot.eef_quat
            utilities.print_log("\nfinal end-effector translation: \n{}".format(contact_eef_pos))
            utilities.print_log("\nfinal end-effector rot_matrix: \n{}".format(T.quat2rotation(contact_eef_quat)))
            
            # Estimate hole pose from model output
            Tgoal_est = np.zeros(7)
            # x, y from top view; z calculated after contact
            Tgoal_est[:2] = T_0B_pos[:2]
            Tgoal_est[2] = contact_eef_pos[2]
            Tgoal_est[3:] = T_0B_quat

            # Check whether the peg has fit into the hole
            # Check x direction
            check_x_success, check_x_time = robot.move_to_contact_request(
            direction=[1, 0, 0, 0, 0, 0],
            speed=0.01,
            force_thresh=10,
            max_duration=2,
            )

            check_x_success = False
            check_y_success = False
            check_z_success = False

            # Check whether the peg has fit into the hole
            # Check x direction
            check_x_success, check_x_time = robot.move_to_contact_request(
            direction=[1, 0, 0, 0, 0, 0],
            speed=0.01,
            force_thresh=10,
            max_duration=1,
            )

            if check_x_success is True:
                # Check y direction
                check_y_success, check_y_time = robot.move_to_contact_request(
                direction=[0, 1, 0, 0, 0, 0],
                speed=0.01,
                force_thresh=10,
                max_duration=1,
                )

                if check_y_success is True:
                    # Check Z direction
                    check_z_success, check_z_time = robot.move_to_contact_request(
                    direction=[0, 0, -1, 0, 0, 0],
                    speed=0.01,
                    force_thresh=10,
                    max_duration=1,
                    )

            if (check_x_success and check_y_success and check_z_success is True):
                insertion_success = True
            else:
                # Insert reinforcement learning manipulation primitives
                args_exp_path = "/home/yeesien/franka_ws/src/square_insertion/policy-square"
                args_config = "/home/yeesien/franka_ws/src/square_insertion/learnmp/exp/config/real-square.json"
                args_log = True
                args_init_pos_noise = 0
                args_init_ori_noise = 0
                args_est_pos_error = 0
                args_est_ori_error = 0
                args_eval_eps = 1

                # ----- TESTING POLICY ONLY
                # Tgoal_true=np.array([0.999304,-0.0333375,-0.0167592,0,-0.0335116,-0.999386,-0.0102189,0,-0.0164082,0.0107734,-0.999807,0,0.63597,0.0264616,0.224734,1]).reshape((4, 4)).T

                insertion_success, insertion_time = evaluate_experiment(args_exp_path, args_config, Tgoal_est, args_log, args_init_pos_noise, args_init_ori_noise, args_est_pos_error, args_est_ori_error, args_eval_eps)

        if insertion_success is True:
            utilities.print_log("Successful insertion!")
            success_count += 1
            utilities.print_log(f'Success rate: {success_count}/{i+1}')
            insertion_end = mytime.time()
            insertion_time_list[i] += insertion_time
            utilities.print_log(f'Insertion time: {insertion_time}')
            utilities.print_log(f'Insertion time list: {insertion_time_list}')
        else:
            utilities.print_log("Unsuccessful insertion!")
            utilities.print_log(f'Success rate: {success_count}/{i+1}')
            insertion_end = mytime.time()
            insertion_time_list[i] += insertion_time
            utilities.print_log(f'Insertion time: {insertion_time}')
            utilities.print_log(f'Insertion time list: {insertion_time_list}')

    utilities.print_log(f'Success rate: {success_count}/50')
    average_model_est_error = np.array(model_est_error)/50

    utilities.print_log(f'Ave model est error: {average_model_est_error}')

    utilities.print_log(f'Insertion time list: {insertion_time_list}')
    utilities.print_log(f'Insertion time mean: {np.mean(insertion_time_list)}')
    utilities.print_log(f'Insertion time std dev: {np.std(insertion_time_list)}')