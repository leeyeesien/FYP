import franka_cartesian_controller.motion_generator as motion_generator
import franka_cartesian_controller.transform_util as T
import random
import math
import capture_image_exe as capture

import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from model import VSNet
from transformations import euler_from_quaternion
from utils import normalize_q, tf_to_quat
from datetime import datetime
import os.path
import time as mytime

robot = motion_generator.MotionGenerator(verbose="debug")
file = "/../logs_ins_exe/log_msgs_{}.txt".format(datetime.now())
i = 0


def print_log(msg):
    print(msg)
    print(msg, file=open(os.path.dirname(__file__) + file, "a"))


def move_to_random_pose(eef_T0C_pos, eef_T0C_quat, pose_id):
    print_log("Moving to random position: T_0{}".format(pose_id))

    r = random.uniform(math.radians(-5), math.radians(5))
    p = random.uniform(math.radians(-5), math.radians(5))
    y = random.uniform(math.radians(-10), math.radians(10))
    eef_TCEi_rot_mat = T.euler2mat([r, p, y])
    eef_TCEi_quat = T.mat2quat(eef_TCEi_rot_mat)
    desired_T0Ei_quat = T.quat_multiply(eef_T0C_quat, eef_TCEi_quat)
    print_log("\nroll: {}". format(math.degrees(r)) +
            "  pitch: {}".format(math.degrees(p)) + "  yaw: {}".format(math.degrees(y)))

    x = random.uniform(-0.005, 0.005)
    y = random.uniform(-0.005, 0.005)
    z = random.uniform(0, 0.01)
    print_log("\nx: {}".format(x) + "  y: {}".format(y) + "  z: {}".format(z))

    robot.move_to_pose_request(
        # random eef position, with new pose origin within a vertical cylinder with r=5 mm & h=10 mm, with the default pose T_0 frame origin at the center of the bottom of the cylinder
        target_pos=eef_T0C_pos + [x, y, z],
        target_quat=desired_T0Ei_quat,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=1
    )

    random_T = []
    random_T[:3] = robot.eef_pos
    random_T[3:] = robot.eef_quat
    print_log("\nend-effector T_0{} random translation: \n{}".format(pose_id, random_T[:3]))
    print_log("\nend-effector T_0{} random rot_matrix: \n{}".format(pose_id, T.quat2rotation(random_T[3:])))

    print_log("Taking picture {}...".format(pose_id))
    image = capture.capture_image(pose_id)

    print_log("T_0{}\n".format(pose_id) + str(random_T[:-1]))
    file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/" + pose_id + ".text", "w")

    if pose_id[0] == "A":
        T_0A = ' '.join([str(elem) for elem in random_T[0:7]])
        print(T_0A, file=file1)
        return image, random_T[0:3], random_T[3:7]
    elif pose_id[0] == "B":
        T_0B = ' '.join([str(elem) for elem in random_T[0:7]])
        print(T_0B, file=file1) 
        return image, random_T[0:3], random_T[3:7]       


def get_relative_pose_T_BA(model, img_A_name, img_B_name, device, img_A, img_B):

    # Find T_BA
    
    # img_A = cv2.imread(os.path.dirname(__file__) + '/../actual_ins_data/img/' + img_A_name + '.png')
    # img_B = cv2.imread(os.path.dirname(__file__) + '/../actual_ins_data/img/' + img_B_name + '.png')

    # Convert img color from BGR to RGB before converting from mat image to PIL
    # img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    # img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

    # Convert mat image to PIL
    # img_A = Image.fromarray(img_A)
    # img_B = Image.fromarray(img_B)

    img_A = Image.open(os.path.dirname(__file__) + '/../ins_exe_data/img/A(ref:T_0A)_i_0.png')
    img_B = Image.open(os.path.dirname(__file__) + '/../ins_exe_data/img/' + img_B_name + '.png')

    # img_A1 = Image.open('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_11012022/peg_round_x0y0_90_2/img/21.png')
    # img_B1 = Image.open('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_11012022/peg_round_x0y0_90_2/img/2.png')
        
    # Convert OpenCV image to Tensor of type float32
    img_transform = transforms.Compose([
        # transforms.Resize(size=(480,640)),
        transforms.ToTensor(),
    ])

    # tf_a = np.loadtxt('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_11012022/peg_round_x0y0_90_2/label/21.txt', dtype=np.float32)
    # tf_b = np.loadtxt('/home/yeesien/franka_ws/src/franka_data_collection/LYS_dataset_11012022/peg_round_x0y0_90_2/label/2.txt', dtype=np.float32)

    # tf_ab = np.matmul(np.linalg.inv(tf_b), tf_a)  # take a as reference
    # label = np.array(tf_to_quat(tf_ab), dtype=np.float32)

    # label = torch.from_numpy(label)
    # label = label.cpu().detach().numpy()

    # print("Label: \n")
    # print(label)
    img_A = img_transform(img_A)
    img_B = img_transform(img_B)

    # img_A1 = img_transform(img_A1)
    # img_B1 = img_transform(img_B1)

    # add batch dimension
    img_A = img_A.unsqueeze(0)
    img_B = img_B.unsqueeze(0)    
    # img_A1 = img_A1.unsqueeze(0)
    # img_B1 = img_B1.unsqueeze(0) 

    img_A = img_A.to(device)
    img_B = img_B.to(device)
    # img_A1 = img_A1.to(device)
    # img_B1 = img_B1.to(device)
    # Find output
    output = model(img_A, img_B)
    output = output.cpu().detach().numpy()
    # output1 = model(img_A1, img_B1)
    # output1 = output1.cpu().detach().numpy()
    print("Output: \n")
    print(output)
    # error = np.zeros(6)

    # print(output[0][0], output1[0][0])
    # x_error = np.abs(output1[0][0] - label[0]) * 1000
    # y_error = np.abs(output1[0][1] - label[1]) * 1000
    # z_error = np.abs(output1[0][2] - label[2]) * 1000

    # xyz_error = np.abs(output[:3] - label[:3]) * 1000
    
    # error[:3] += xyz_error                    

    # error[:2] += np.abs(output[j, :2] - label[j, :2]
    # quat_output = normalize_q(output[0][3:])
    # quat_label = label[3:]
    # quat_output1 = normalize_q(output1[0][3:])

    # rpy_output = np.array(euler_from_quaternion(quat_output))
    # rpy_output1 = np.array(euler_from_quaternion(quat_output1))
    # rpy_label = np.array(euler_from_quaternion(quat_label))
    # rpy_error = np.rad2deg(np.abs(rpy_output1 - rpy_label))
    # error[3:] += rpy_error
    # print(error)
    return output[0][:3], normalize_q(output[0][3:])


def move_to_T_0A_est(est_pos, est_quat):
    robot.move_to_pose_request(
        target_pos=est_pos,
        target_quat=est_quat,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
    )
    achieved_T_0A_pos_est = robot.eef_pos
    achieved_T_0A_quat_est = robot.eef_quat
    return achieved_T_0A_pos_est, achieved_T_0A_quat_est


def find_errors_between_2_poses(pose1_pos, pose1_quat, pose2_pos, pose2_quat):
   
    # Find position errors between pose 1 and pose 2
    x_error = np.abs(pose1_pos[0] - pose2_pos[0]) * 1000
    y_error = np.abs(pose1_pos[1] - pose2_pos[1]) * 1000
    z_error = np.abs(pose1_pos[2] - pose2_pos[2]) * 1000

    # Find rotation errors (roll, pitch, yaw) between pose 1 and pose 2
    ref_rpy = np.array(euler_from_quaternion(pose2_quat))
    est_rpy = np.array(euler_from_quaternion(pose1_quat))

    # Convert r, p, y angles to between -pi and pi
    for i in range(len(ref_rpy)):
        if ref_rpy[i] > math.pi:
            ref_rpy[i] -= (2 * math.pi)
        elif ref_rpy[i] < -math.pi:
            ref_rpy[i] += (2 * math.pi)

    for i in range(len(est_rpy)):
        if est_rpy[i] > math.pi:
            est_rpy[i] -= (2 * math.pi)
        elif est_rpy[i] < -math.pi:
            est_rpy[i] += (2 * math.pi)

    # print("ref_rpy: \n{}\n".format(T_0A_ref_rpy))
    # print("est_rpy: \n{}\n".format(T_0A_est_rpy))

    roll_error = np.rad2deg(np.abs(est_rpy[0] - ref_rpy[0]))
    if roll_error > 180:
        roll_error = np.abs(roll_error - 360)

    pitch_error = np.rad2deg(np.abs(est_rpy[1] - ref_rpy[1]))
    if pitch_error > 180:
        pitch_error = np.abs(pitch_error - 360)

    yaw_error = np.rad2deg(np.abs(est_rpy[2] - ref_rpy[2]))
    if yaw_error > 180:
        yaw_error = np.abs(yaw_error - 360)

    return x_error, y_error, z_error, roll_error, pitch_error, yaw_error


if __name__ == "__main__":

    starting_eef_pos = robot.eef_pos
    starting_eef_quat = robot.eef_quat
    print_log("\nend-effector starting translation wrt fixed base frame: \n{}".format(starting_eef_pos))
    print_log("\nend-effector starting rot_matrix wrt fixed base frame: \n{}".format(T.quat2rotation(starting_eef_quat)))

    device = torch.device('cpu')

    model2 = torch.load('/home/yeesien/franka_ws/src/square_insertion/VSNet/YS_VSNet-train_10cm_square_28022022_log_everything-200^2*20/model.pth', map_location=device)

    model = VSNet(num_classes=7)

    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(model2.module.state_dict())

    model.eval()
    model.to(device)

    # robot.move_to_pose_request(
    # target_pos=[starting_eef_pos[0], starting_eef_pos[1], 0.37],    # go to T0 (15 cm above hole)
    # linear_speed=0.1,
    # rotation_speed=0.4,
    # max_duration=5
    # )
    T0C_pos = [robot.eef_pos[0], robot.eef_pos[1], 0.34]
    rotm_0C = np.zeros((3, 3), dtype=float)
    rotm_0C[0, 0] = 0.9626807 
    rotm_0C[0, 1] = -0.26862115
    rotm_0C[0, 2] = -0.03299282
    rotm_0C[1, 0] = -0.2678331  
    rotm_0C[1, 1] = -0.96310264
    rotm_0C[1, 2] = 0.026429297
    rotm_0C[2, 0] = -0.038874943  
    rotm_0C[2, 1] = -0.016606404
    rotm_0C[2, 2] = -0.999106
    T0C_quat = T.mat2quat(rotm_0C)

    robot.move_to_pose_request(
        target_pos=T0C_pos,
        target_quat=T0C_quat,
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
    )

    # eef_T0C_pos = T0C_pos
    # eef_T0C_quat = T0C_quat
    # print("\nend-effector T_0C translation: \n{}".format(eef_T0C_pos))
    # print("\nend-effector T_0C rot_matrix: \n{}".format(T.quat2rotation(robot.eef_quat)))

    # TCB_pos = [-0.00047171116, -0.0022984035, -0.008955359]
    # rotm = np.zeros((3, 3), dtype=float)
    # rotm[0, 0] = 0.99966705   
    # rotm[0, 1] = -0.016333742
    # rotm[0, 2] = 0.019976402 
    # rotm[1, 0] = 0.016524559  
    # rotm[1, 1] = 0.99981904
    # rotm[1, 2] = -0.00942468  
    # rotm[2, 0] = -0.019818848  
    # rotm[2, 1] = 0.009751643
    # rotm[2, 2] = 0.99975604  
    # TCB_quat = T.mat2quat(rotm)
    # end_pos, end_quat = T.multiply_pose(eef_T0C_pos, eef_T0C_quat, TCB_pos, TCB_quat)

    # robot.move_to_pose_request(
    #     target_pos=end_pos,
    #     target_quat=end_quat,
    #     linear_speed=0.1,
    #     rotation_speed=0.4,
    #     max_duration=5
    # )

    # print_log("Taking picture {}...".format(i))
    # image_T_0A_ref = capture.capture_image("A(ref:T_0A)_i_{}".format(i))

    # TCB2_pos = [-0.0024878383, -0.0016901623, -0.00411427]
    # rotm2 = np.zeros((3, 3), dtype=float)
    # rotm2[0, 0] = 0.9964945     
    # rotm2[0, 1] = 0.08352803
    # rotm2[0, 2] = -0.004668721 
    # rotm2[1, 0] = -0.08320916  
    # rotm2[1, 1] = 0.99537444
    # rotm2[1, 2] = 0.04802092  
    # rotm2[2, 0] = 0.008658219  
    # rotm2[2, 1] = -0.047464103
    # rotm2[2, 2] = 0.99883544  
    # TCB2_quat = T.mat2quat(rotm2)
    # end_pos2, end_quat2 = T.multiply_pose(eef_T0C_pos, eef_T0C_quat, TCB2_pos, TCB2_quat)

    # robot.move_to_pose_request(
    #     target_pos=end_pos2,
    #     target_quat=end_quat2,
    #     linear_speed=0.1,
    #     rotation_speed=0.4,
    #     max_duration=5
    # )

    # print_log("Taking picture {}...".format(i))
    # image_T_0B = capture.capture_image("B(test:T_0B)_i_{}".format(i))

    # T_BA_pos, T_BA_quat = get_relative_pose_T_BA(model, "A(ref:T_0A)_i_{}".format(i), "B(test:T_0B)_i_{}".format(i), device, image_T_0A_ref, image_T_0B)

    # robot.move_to_pose_request(
    # target_pos=[starting_eef_pos[0], starting_eef_pos[1], 0.37],    # go to T0 (15 cm above hole)
    # linear_speed=0.1,
    # rotation_speed=0.4,
    # max_duration=5
    # )

    eef_T0C_pos = robot.eef_pos
    eef_T0C_quat = robot.eef_quat
    print_log("\nend-effector T_0C translation: \n{}".format(eef_T0C_pos))
    print_log("\nend-effector T_0C rot_matrix: \n{}".format(T.quat2rotation(eef_T0C_quat)))

    # est_errors = np.zeros(6)
    # controller_errors = np.zeros(6)

    print_log("\nAttempt {}\n".format(i+1))
    T_0A_pos_ref = eef_T0C_pos
    T_0A_quat_ref = eef_T0C_quat

    print_log("\nend-effector T_0A(ref:T_0A)_i_{} translation: \n{}".format(i, T_0A_pos_ref))
    print_log("\nend-effector T_0A(ref:T_0A)_i_{} rot_matrix: \n{}".format(i, T.quat2rotation(T_0A_quat_ref)))

    print_log("Taking picture {}...".format(i))
    image_T_0A_ref = capture.capture_image("A(ref:T_0A)_i_{}".format(i))

    T_0A_ref = []
    T_0A_ref[:3] = T_0A_pos_ref
    T_0A_ref[3:] = T_0A_quat_ref

    print_log("T_0A(ref:T_0A)_i_{}\n".format(i) + str(T_0A_ref[:-1]))
    file1 = open(os.path.dirname(__file__) + "/../ins_exe_data/label/A(ref:T_0A)_i_" + str(i) + ".text", "w")

    T_0A = ' '.join([str(elem) for elem in T_0A_ref[0:7]])
    print(T_0A, file=file1)

    success_count = 0
    model_est_error = np.zeros(6)
    controller_error = np.zeros(6)
    final_error = np.zeros(6)

    for i in range(50):

        if i >= 1:
            robot.move_to_pose_request(
                target_pos=eef_T0C_pos,
                target_quat=eef_T0C_quat,
                linear_speed=0.1,
                rotation_speed=0.4,
                max_duration=5
            )
        # Move to a random pose within the cylinder
        image_T_0B, T_0B_pos, T_0B_quat = move_to_random_pose(eef_T0C_pos, eef_T0C_quat, "B(test:T_0B)_i_{}".format(i))

        # inverse_T_0B = T.pose_inverse(T_0B_pos, T_0B_quat)

        # T_BA_pos_actual, T_BA_quat_actual = T.multiply_pose(inverse_T_0B[0], inverse_T_0B[1], T_0A_pos_ref, T_0A_quat_ref)

        T_BA_pos, T_BA_quat = get_relative_pose_T_BA(model, "A(ref:T_0A)_i_0".format(i), "B(test:T_0B)_i_{}".format(i), device, image_T_0A_ref, image_T_0B)

        T_0A_pos_est, T_0A_quat_est = T.multiply_pose(T_0B_pos, T_0B_quat, T_BA_pos, T_BA_quat)
        print_log("Estimated T_0A_pos: \n{}; \nEstimated T_0A_quat: \n{}".format(T_0A_pos_est, T_0A_quat_est))

        # Find errors between T_0A_ref and T_0A_est
        T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error = find_errors_between_2_poses(T_0A_pos_est, T_0A_quat_est, T_0A_pos_ref, T_0A_quat_ref)

        # print_log all the errors above (model error)
        print_log("Model estimation errors T_0A: x = {} mm, y = {} mm, z = {} mm, r = {} deg, pitch = {} deg, yaw = {} deg".format(T_0A_x_error, T_0A_y_error, T_0A_z_error, T_0A_roll_error, T_0A_pitch_error, T_0A_yaw_error))

        model_est_error[0] += T_0A_x_error 
        model_est_error[1] += T_0A_y_error
        model_est_error[2] += T_0A_z_error
        model_est_error[3] += T_0A_roll_error
        model_est_error[4] += T_0A_pitch_error
        model_est_error[5] += T_0A_yaw_error

        # Move to estimated T_0A
        achieved_T_0A_pos_est, achieved_T_0A_quat_est = move_to_T_0A_est(T_0A_pos_est, T_0A_quat_est)

        mytime.sleep(1)

        # get achieved_T_0A_pos_quat_est and compare it to T_0A_pos_quat_est, the difference is controller error
        controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error = find_errors_between_2_poses(achieved_T_0A_pos_est, achieved_T_0A_quat_est, T_0A_pos_est, T_0A_quat_est)

        # print_log controller errors
        print_log("Controller errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(controller_x_error, controller_y_error, controller_z_error, controller_roll_error, controller_pitch_error, controller_yaw_error))

        controller_error[0] += controller_x_error 
        controller_error[1] += controller_y_error
        controller_error[2] += controller_z_error
        controller_error[3] += controller_roll_error
        controller_error[4] += controller_pitch_error
        controller_error[5] += controller_yaw_error

        robot.move_to_pose_request(
        target_pos=achieved_T_0A_pos_est - [0, 0, 0.10],    # peg-in-hole insertion
        linear_speed=0.1,
        rotation_speed=0.4,
        max_duration=5
        )

        success, time = robot.move_to_contact_request(
            direction=[0, 0, -1, 0, 0, 0],
            speed=0.01,
            force_thresh=10,
            max_duration=20,
        )

        print_log("Move to contact request success and time: \n")
        print_log(success)
        print(time)

        final_eef_pos = robot.eef_pos
        final_eef_quat = robot.eef_quat
        print_log("\nfinal end-effector translation: \n{}".format(final_eef_pos))
        print_log("\nfinal end-effector rot_matrix: \n{}".format(T.quat2rotation(final_eef_quat)))

        # Find errors between starting position and final position after visual servoing
        final_x_error, final_y_error, final_z_error, final_roll_error, final_pitch_error, final_yaw_error = find_errors_between_2_poses(final_eef_pos, final_eef_quat, starting_eef_pos, starting_eef_quat)
        
        # print_log final(before/after) errors
        print_log("Final errors: x = {} mm, y = {} mm, z = {} mm, roll = {} deg, pitch = {} deg, yaw = {} deg".format(final_x_error, final_y_error, final_z_error, final_roll_error, final_pitch_error, final_yaw_error))

        final_error[0] += final_x_error 
        final_error[1] += final_y_error
        final_error[2] += final_z_error
        final_error[3] += final_roll_error
        final_error[4] += final_pitch_error
        final_error[5] += final_yaw_error

        if success is True:
            print("Transformation matrix after moving down to contact: \n")
            print(robot.eef_pos)
            success, time = robot.insert_by_admittance_request(
                admittance_gain = [0.01, 0.01, 0.01] + [0.]*3, # good translational
                target_pos=starting_eef_pos,
                max_duration=10,
                target_force=[0, 0, -12, 0, 0, 0],
            )
        
            print_log("Insert by admittance request success and time: \n")
            print_log(success)
            print_log(time)
            if success is True:
                success_count += 1

    print(f'Success rate: {success_count}/50')
    average_model_est_error = np.array(model_est_error)/50
    average_controller_error = np.array(controller_error)/50
    average_final_error = np.array(final_error)/50

    print(f'Ave model est error: {average_model_est_error}, Ave controller error: {average_controller_error}, Ave final error: {average_final_error}')