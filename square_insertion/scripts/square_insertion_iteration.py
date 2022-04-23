import franka_cartesian_controller.motion_generator as motion_generator
import franka_cartesian_controller.transform_util as T
from peg_insertion_functions import InsertionMotions, ModelEstimation, Utilities

import torch
import numpy as np
from model import VSNet
from datetime import datetime
import os.path
import time as mytime

from eval_real_square import evaluate_experiment
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

NUM_ITER = 10

if __name__ == "__main__":
    robot = motion_generator.MotionGenerator(verbose="debug")
    insertion_motions = InsertionMotions()
    model_estimation = ModelEstimation()
    utilities = Utilities()

    file = "/../logs_ins_exe/log_msgs_{}.txt".format(datetime.now())
    i = 0

    starting_eef_pos = robot.eef_pos
    starting_eef_quat = robot.eef_quat
    utilities.print_log("\nend-effector starting translation wrt fixed base frame: \n{}".format(starting_eef_pos))
    utilities.print_log("\nend-effector starting rot_matrix wrt fixed base frame: \n{}".format(T.quat2rotation(starting_eef_quat)))

    # Load model
    device = torch.device('cpu')
    model2 = torch.load('/home/yeesien/franka_ws/src/square_insertion/VSNet/YS_VSNet-train_10cm_square_28022022_log_everything-200^2*20/model.pth', map_location=device)
    model = VSNet(num_classes=7)
    model.load_state_dict(model2.module.state_dict())
    model.eval()
    model.to(device)

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

    # Capture goal image
    utilities.print_log("Taking picture {}...".format(i))
    image_T_0A_ref = utilities.capture_image("A(ref:T_0A)_i_{}".format(i))

    # Log true goal pose transformation matrix
    T_0A_ref = []
    T_0A_ref[:3] = T_0A_pos_ref
    T_0A_ref[3:] = T_0A_quat_ref
    utilities.print_log("T_0A(ref:T_0A)_i_{}\n".format(i) + str(T_0A_ref))
    file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/A(ref:T_0A)_i_" + str(i) + ".text", "w")
    T_0A = ' '.join([str(elem) for elem in T_0A_ref[0:7]])
    print(T_0A, file=file1)

    # success_count = 0
    # model_est_error = np.zeros(6)
    # controller_error = np.zeros(6)
    rows, columns = (NUM_ITER, 6)
    est_errors_iterations = [[0]*columns for _ in range(rows)]
    # pre_align_time_list = [0] * 50
    # insertion_time_list = [0] * 50
    # total_time_list = [0] * 50

    # for i in range(1):
    #     if i >= 1:
    #         robot.move_to_pose_request(
    #             target_pos=T_0A_pos_ref,
    #             target_quat=T_0A_quat_ref,
    #             linear_speed=0.1,
    #             rotation_speed=0.4,
    #             max_duration=5
    #         )

    utilities.print_log("\nAttempt {}\n".format(i+1))

    # Move to a random pose within the cylinder
    image_T_0B, T_0B_pos, T_0B_quat, pre_align_start = insertion_motions.move_to_big_sampling_error(T_0A_pos_ref, T_0A_quat_ref, "B(test:T_0B)_i_{}".format(i))

    for iteration in range(NUM_ITER):
        T_BA_pos, T_BA_quat = model_estimation.get_relative_pose_T_BA(model, "A(ref:T_0A)_i_0".format(i), "B(test:T_0B)_i_{}".format(iteration), device, image_T_0A_ref, image_T_0B)

        # quat_Bold_B = T.axisangle2quat([0, 0, 1.08])
        # quat_B_Bold = T.quat_inverse(quat_Bold_B)
        # rot_mat_B_Bold = T.quat2rotation(quat_B_Bold)
        # T_BA_pos = np.matmul(rot_mat_B_Bold, T_BA_pos.transpose())

        T_0A_pos_est, T_0A_quat_est = T.multiply_pose(T_0B_pos, T_0B_quat, T_BA_pos, T_BA_quat)
        utilities.print_log("Estimated T_0A_pos: \n{}; \nEstimated T_0A_quat: \n{}".format(T_0A_pos_est, T_0A_quat_est))

        # Find errors between T_0A_ref and T_0A_est
        T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error = utilities.find_errors_between_2_poses(T_0A_pos_est, T_0A_quat_est, T_0A_pos_ref, T_0A_quat_ref)

        # utilities.print_log all the errors above (model error)
        utilities.print_log("Model estimation errors T_0A: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error))

        est_errors_iterations[iteration][0] += T_0A_x_error
        est_errors_iterations[iteration][1] += T_0A_y_error
        est_errors_iterations[iteration][2] += T_0A_z_error
        est_errors_iterations[iteration][3] += T_0A_roll_error
        est_errors_iterations[iteration][4] += T_0A_pitch_error
        est_errors_iterations[iteration][5] += T_0A_yaw_error

        # If all errors are smaller than 1.5 mm or 1.5 deg, stop iteration
        if (all(element < 1.5 for element in est_errors_iterations[iteration])):
            NUM_ITER = iteration+1
            break 
        # if (all(element < 1.5 for element in est_errors_iterations[iteration][0:2]) and all(element < 1.5 for element in est_errors_iterations[iteration][3:7])):
        #     NUM_ITER = iteration+1
        #     break 

        # Move to estimated T_0A
        # achieved_T_0A_pos_est, achieved_T_0A_quat_est = insertion_motions.move_to_T_0A_est(T_0A_pos_est, T_0A_quat_est)
        T_0B_pos, T_0B_quat = insertion_motions.move_to_T_0A_est(T_0A_pos_est, T_0A_quat_est)

        mytime.sleep(1)

        utilities.capture_image("B(test:T_0B)_i_{}".format(iteration+1))

        # # get achieved_T_0A_pos_quat_est and compare it to T_0A_pos_quat_est, the difference is controller error
        # controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error = utilities.find_errors_between_2_poses(achieved_T_0A_pos_est, achieved_T_0A_quat_est, T_0A_pos_est, T_0A_quat_est)

        # # utilities.print_log controller errors
        # utilities.print_log("Controller errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error))

        # model_est_error[0] += T_0A_x_error 
        # model_est_error[1] += T_0A_y_error
        # model_est_error[2] += T_0A_z_error
        # model_est_error[3] += T_0A_roll_error
        # model_est_error[4] += T_0A_pitch_error
        # model_est_error[5] += T_0A_yaw_error

        # controller_error[0] += controller_x_error 
        # controller_error[1] += controller_y_error
        # controller_error[2] += controller_z_error
        # controller_error[3] += controller_roll_error
        # controller_error[4] += controller_pitch_error
        # controller_error[5] += controller_yaw_error

    x_iterations_array = np.array([0] * (NUM_ITER + 1))
    x_iterations_array[0] = 0
    for iterations in range(NUM_ITER):
        x_iterations_array[iterations+1] = int(iterations + 1)
    
    rows, columns = (NUM_ITER+1, 6)
    y_average_est_errors_iterations = [[0]*columns for _ in range(rows)]
    y_average_est_errors_iterations[0] = [10, 10, 10, 10, 10, 20]
    y_average_est_errors_iterations[1:NUM_ITER+1] = est_errors_iterations[0:NUM_ITER]

    utilities.print_log("Error list: ")
    utilities.print_log(y_average_est_errors_iterations)

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
        utilities.print_log("\ncontact end-effector translation: \n{}".format(contact_eef_pos))
        utilities.print_log("\ncontact end-effector rot_matrix: \n{}".format(T.quat2rotation(contact_eef_quat))) 

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

        if ((check_x_success and check_y_success and check_z_success is True) and (robot.eef_pos[2]-starting_eef_pos[2]) < 0.004):
            insertion_time = 0
            insertion_success = True
        else:
            # Estimate hole pose from model output
            Tgoal_est = np.zeros(7)
            # x, y from top view; z calculated after contact
            Tgoal_est[:2] = T_0A_pos_est[:2]
            Tgoal_est[2] = contact_eef_pos[2]
            Tgoal_est[3:] = T_0A_quat_est   

            # Insert reinforcement learning manipulation primitives
            args_exp_path = "/home/yeesien/franka_ws/src/square_insertion/policy-square"
            args_config = "/home/yeesien/franka_ws/src/square_insertion/learnmp/exp/config/real-square.json"
            args_log = True
            args_init_pos_noise = 0
            args_init_ori_noise = 0
            args_est_pos_error = 0
            args_est_ori_error = 0
            args_eval_eps = 1
            insertion_success, insertion_time = evaluate_experiment(args_exp_path, args_config, Tgoal_est, args_log, args_init_pos_noise, args_init_ori_noise, args_est_pos_error, args_est_ori_error, args_eval_eps)

        # if insertion_success is True:
        #     insertion_end = mytime.time()
        #     utilities.print_log("Successful insertion!")
        #     success_count += 1
        #     utilities.print_log(f'Success rate: {success_count}/{i+1}')
        #     pre_align_time = pre_align_end - pre_align_start
        #     pre_align_time_list[i] += pre_align_time
        #     insertion_time_list[i] += insertion_time
        #     total_time_list[i] += (pre_align_time + insertion_time)
        #     utilities.print_log(f'Pre-alignment time: {pre_align_time}')
        #     utilities.print_log(f'Insertion time: {insertion_time}')
        #     utilities.print_log(f'Total time: {pre_align_time + insertion_time}')
        #     utilities.print_log(f'Pre-alignment time list: {pre_align_time_list}')
        #     utilities.print_log(f'Insertion time list: {insertion_time_list}')
        #     utilities.print_log(f'Total time list: {total_time_list}')
        # else:
        #     insertion_end = mytime.time()
        #     utilities.print_log("Unsuccessful insertion!")
        #     utilities.print_log(f'Success rate: {success_count}/{i+1}')
        #     pre_align_time = pre_align_end - pre_align_start
        #     pre_align_time_list[i] += pre_align_time
        #     insertion_time_list[i] += insertion_time
        #     total_time_list[i] += (pre_align_time + insertion_time)
        #     utilities.print_log(f'Pre-alignment time: {pre_align_time}')
        #     utilities.print_log(f'Insertion time: {insertion_time}')
        #     utilities.print_log(f'Total time: {pre_align_time + insertion_time}')
        #     utilities.print_log(f'Pre-alignment time list: {pre_align_time_list}')
        #     utilities.print_log(f'Insertion time list: {insertion_time_list}')
        #     utilities.print_log(f'Total time list: {total_time_list}')

    # # Smoothen the curves
    # x_y_spline_0 = make_interp_spline(x_iterations_array, np.array([y[0] for y in y_average_est_errors_iterations]))
    # x0_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
    # y0_ = x_y_spline_0(x0_)

    # x_y_spline_1 = make_interp_spline(x_iterations_array, np.array([y[1] for y in y_average_est_errors_iterations]))
    # x1_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
    # y1_ = x_y_spline_1(x1_)

    # x_y_spline_2 = make_interp_spline(x_iterations_array, np.array([y[2] for y in y_average_est_errors_iterations]))
    # x2_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
    # y2_ = x_y_spline_2(x2_)

    # x_y_spline_3 = make_interp_spline(x_iterations_array, np.array([y[3] for y in y_average_est_errors_iterations]))
    # x3_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
    # y3_ = x_y_spline_3(x3_)

    # x_y_spline_4 = make_interp_spline(x_iterations_array, np.array([y[4] for y in y_average_est_errors_iterations]))
    # x4_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
    # y4_ = x_y_spline_4(x4_)

    # x_y_spline_5 = make_interp_spline(x_iterations_array, np.array([y[5] for y in y_average_est_errors_iterations]))
    # x5_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
    # y5_ = x_y_spline_5(x5_)

    # # Plot curves
    # figure, axis = plt.subplots(2, 1)

    # # Translation errors curves
    # axis[0].plot(x0_, y0_, color='r', label='x errors')
    # axis[0].plot(x1_, y1_, color='g', label='y errors')
    # axis[0].plot(x2_, y2_, color='b', label='z errors')
    # axis[0].set_title("Translation errors")
    # axis[0].set_xlabel("No. of Iterations")
    # axis[0].set_xticks(np.arange(min(x_iterations_array), max(x_iterations_array)+1, 1.0))
    # axis[0].set_ylabel("Errors (in mm)")
    # axis[0].legend(loc='best')

    # # Orientation errors curves
    # axis[1].plot(x3_, y3_, color='r', label='roll errors')
    # axis[1].plot(x4_, y4_, color='r', label='pitch errors')
    # axis[1].plot(x5_, y5_, color='r', label='yaw errors')
    # axis[1].set_title("Orientation errors")
    # axis[1].set_xlabel("No. of Iterations")
    # axis[1].set_xticks(np.arange(min(x_iterations_array), max(x_iterations_array)+1, 1.0))
    # axis[1].set_ylabel("Errors (in deg)")
    # axis[1].legend(loc='best')

    # plt.show()

        # pre_align_end = mytime.time()

        # contact_success, contact_time = robot.move_to_contact_request(
        #     direction=[0, 0, -1, 0, 0, 0],
        #     speed=0.01,
        #     force_thresh=10,
        #     max_duration=20,
        # )

        # utilities.print_log("Move to contact request success and time: \n")
        # utilities.print_log(contact_success)
        # utilities.print_log(contact_time)

        # if contact_success is True:
        #     contact_eef_pos = robot.eef_pos
        #     contact_eef_quat = robot.eef_quat
        #     utilities.print_log("\ncontact end-effector translation: \n{}".format(contact_eef_pos))
        #     utilities.print_log("\ncontact end-effector rot_matrix: \n{}".format(T.quat2rotation(contact_eef_quat))) 

        #     check_x_success = False
        #     check_y_success = False
        #     check_z_success = False

        #     # Check whether the peg has fit into the hole
        #     # Check x direction
        #     check_x_success, check_x_time = robot.move_to_contact_request(
        #     direction=[1, 0, 0, 0, 0, 0],
        #     speed=0.01,
        #     force_thresh=10,
        #     max_duration=1,
        #     )

        #     if check_x_success is True:
        #         # Check y direction
        #         check_y_success, check_y_time = robot.move_to_contact_request(
        #         direction=[0, 1, 0, 0, 0, 0],
        #         speed=0.01,
        #         force_thresh=10,
        #         max_duration=1,
        #         )

        #         if check_y_success is True:
        #             # Check Z direction
        #             check_z_success, check_z_time = robot.move_to_contact_request(
        #             direction=[0, 0, -1, 0, 0, 0],
        #             speed=0.01,
        #             force_thresh=10,
        #             max_duration=1,
        #             ) 

        #     if ((check_x_success and check_y_success and check_z_success is True) and (robot.eef_pos[2]-starting_eef_pos[2]) < 0.004):
        #         insertion_time = 0
        #         insertion_success = True
        #     else:
        #         # Estimate hole pose from model output
        #         Tgoal_est = np.zeros(7)
        #         # x, y from top view; z calculated after contact
        #         Tgoal_est[:2] = T_0A_pos_est[:2]
        #         Tgoal_est[2] = contact_eef_pos[2]
        #         Tgoal_est[3:] = T_0A_quat_est   

        #         # Insert reinforcement learning manipulation primitives
        #         args_exp_path = "/home/yeesien/franka_ws/src/square_insertion/policy-square"
        #         args_config = "/home/yeesien/franka_ws/src/square_insertion/learnmp/exp/config/real-square.json"
        #         args_log = True
        #         args_init_pos_noise = 0
        #         args_init_ori_noise = 0
        #         args_est_pos_error = 0
        #         args_est_ori_error = 0
        #         args_eval_eps = 1
        #         insertion_success, insertion_time = evaluate_experiment(args_exp_path, args_config, Tgoal_est, args_log, args_init_pos_noise, args_init_ori_noise, args_est_pos_error, args_est_ori_error, args_eval_eps)

        #     if insertion_success is True:
        #         insertion_end = mytime.time()
        #         utilities.print_log("Successful insertion!")
        #         success_count += 1
        #         utilities.print_log(f'Success rate: {success_count}/{i+1}')
        #         pre_align_time = pre_align_end - pre_align_start
        #         pre_align_time_list[i] += pre_align_time
        #         insertion_time_list[i] += insertion_time
        #         total_time_list[i] += (pre_align_time + insertion_time)
        #         utilities.print_log(f'Pre-alignment time: {pre_align_time}')
        #         utilities.print_log(f'Insertion time: {insertion_time}')
        #         utilities.print_log(f'Total time: {pre_align_time + insertion_time}')
        #         utilities.print_log(f'Pre-alignment time list: {pre_align_time_list}')
        #         utilities.print_log(f'Insertion time list: {insertion_time_list}')
        #         utilities.print_log(f'Total time list: {total_time_list}')
        #     else:
        #         insertion_end = mytime.time()
        #         utilities.print_log("Unsuccessful insertion!")
        #         utilities.print_log(f'Success rate: {success_count}/{i+1}')
        #         pre_align_time = pre_align_end - pre_align_start
        #         pre_align_time_list[i] += pre_align_time
        #         insertion_time_list[i] += insertion_time
        #         total_time_list[i] += (pre_align_time + insertion_time)
        #         utilities.print_log(f'Pre-alignment time: {pre_align_time}')
        #         utilities.print_log(f'Insertion time: {insertion_time}')
        #         utilities.print_log(f'Total time: {pre_align_time + insertion_time}')
        #         utilities.print_log(f'Pre-alignment time list: {pre_align_time_list}')
        #         utilities.print_log(f'Insertion time list: {insertion_time_list}')
        #         utilities.print_log(f'Total time list: {total_time_list}')

    # utilities.print_log(f'\nSuccess rate: {success_count}/50')
    # average_model_est_error = np.array(model_est_error)/50
    # average_controller_error = np.array(controller_error)/50

    # utilities.print_log(f'Ave model est error: {average_model_est_error}, Ave controller error: {average_controller_error}')

    # utilities.print_log(f'Pre-alignment time list: {pre_align_time_list}')
    # utilities.print_log(f'Pre-alignment time mean: {np.mean(pre_align_time_list)}')
    # utilities.print_log(f'Pre-alignment time std dev: {np.std(pre_align_time_list)}')

    # utilities.print_log(f'Insertion time list: {insertion_time_list}')
    # utilities.print_log(f'Insertion time mean: {np.mean(insertion_time_list)}')
    # utilities.print_log(f'Insertion time std dev: {np.std(insertion_time_list)}')

    # utilities.print_log(f'Total time list: {total_time_list}')
    # utilities.print_log(f'Total time mean: {np.mean(total_time_list)}')
    # utilities.print_log(f'Total time std dev: {np.std(total_time_list)}')