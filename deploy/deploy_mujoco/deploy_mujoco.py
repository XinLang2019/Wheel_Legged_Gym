import time

import mujoco.viewer
import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import onnxruntime as ort
import matplotlib.pyplot as plt

DEPLOY_DIR = '/home/zy/work/Wheel_Legged_Gym'


class Deploy:
    def __init__(self):
        import argparse

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("config_file", type=str, help="config file name in the config folder")
        self.args = self.parser.parse_args()
        self.config_file = self.args.config_file
        with open(f"{DEPLOY_DIR}/deploy/deploy_mujoco/configs/{self.config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = config["policy_path"]
            self.xml_path = config["xml_path"]
            print(self.xml_path)
            print(self.policy_path)

            self.simulation_duration = config["simulation_duration"]
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.torque_limits = np.array(config["torque_limits"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.vel_scale = config["vel_scale"]

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.wheel_indices = config["wheel_sim_indices"]
            self.cmd = np.array(config["cmd_init"], dtype=np.float32)

        # define context variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs_hist_buf = np.zeros(self.num_obs * 5, dtype=np.float32)

        self.dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.line_vel = np.zeros(3, dtype=np.float32)

        self.counter = 0

        # Load robot model
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt

        # load policy
        # self.policy = torch.jit.load(self.policy_path)
        self.policy = ort.InferenceSession(self.policy_path, provifers=["CPUExecutionProvider"])
    
    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    
    def get_robot_state(self):
        self.dof_pos = self.d.qpos[7:]
        self.dof_vel = self.d.qvel[6:]
        self.quat = self.d.qpos[3:7]
        self.ang_vel = self.d.qvel[3:6]
        self.line_vel = self.d.qvel[3:6]
    
    def compute_torques(self, actions):
        dof_err = self.default_angles - self.dof_pos # 各DOF默认位置 - 目前各DOF位置
        dof_err[self.wheel_indices] =  0 # 轮子的误差是0

        actions_scaled = actions * self.action_scale # action * 0.25
        actions_scaled[self.wheel_indices] = 0 # 轮子使用速度控制，角度增量为0
        vel_ref = np.zeros_like(actions_scaled)
        vel_tmp = actions * self.vel_scale # action提供期望速度
        vel_ref[self.wheel_indices] = vel_tmp[self.wheel_indices] # 只有轮子使用速度控制
        
        
        torques = self.kps * (
                actions_scaled + dof_err
            ) + self.kds * (vel_ref - self.dof_vel)
        
    
        return np.clip(torques, -self.torque_limits, self.torque_limits)
    
    def compute_observation(self):
        dof_error = (self.dof_pos - self.default_angles) * self.dof_pos_scale
        dof_error[self.wheel_indices] = 0

        dof_pos = self.dof_pos.copy()
        dof_pos[self.wheel_indices] = 0

        dof_vel = self.dof_vel * self.dof_vel_scale
        gravity_orientation = self.get_gravity_orientation(self.quat)
        omega = self.ang_vel * self.ang_vel_scale
        
        self.obs[:3] = omega
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9 : 9 + self.num_actions] = dof_error
        self.obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dof_vel
        self.obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = dof_pos
        self.obs[9 + 3 * self.num_actions : 9 + 4 * self.num_actions] = self.action
        
        self.obs_hist_buf = self.obs_hist_buf[73:]
        self.obs_hist_buf = np.concatenate((self.obs_hist_buf, self.obs), axis=-1)

    def plot_data(self, data1, data2=None):
        fig, axs = plt.subplots(4, 4, figsize=(15, 10))
        axs = axs.flatten()
           
        data1 = np.array(data1)
        # data2 = np.array(data2)
        print(data1.shape)
        for i in range(16):
            axs[i].plot(data1[:, i])
            # axs[i].plot(data2[:, i])
        plt.show()

    def run_sim(self):
        self.vel_data = []
        self.cmd_data = []
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            
            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()
                tau = self.compute_torques(self.action)
                self.d.ctrl[:] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(self.m, self.d)

                self.counter += 1
                if self.counter % self.control_decimation == 0:
                    self.get_robot_state()
                    self.compute_observation()
                    
                    out_name = self.policy.get_outputs()[0].name
                    actions = self.policy.run([out_name], {"obs": self.obs.reshape(1, -1), "obs_hist":self.obs_hist_buf.reshape(1, -1)})
                    self.action = actions[0][0]

                    self.vel_data.append(self.dof_pos.copy())
                    # self.cmd_data.append(self.cmd)


                    # self.action = np.zeros_like(self.action)
                    # print(self.action.shape)
                if self.counter == 5000:
                    self.plot_data(self.vel_data)
                    
                viewer.sync()
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    deploy = Deploy()

    deploy.run_sim()
