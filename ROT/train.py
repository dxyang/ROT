#!/usr/bin/env python3

import warnings
import os

# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import copy
from datetime import datetime
import sys
curr_file = Path.cwd()
rewardlearning_vid_repo_root = "/home/dxyang/code/rewardlearning-vid"
sys.path.append(rewardlearning_vid_repo_root)
from policy_learning.envs import ImageMetaworldEnv
from reward_extraction.data import H5PyTrajDset

import hydra
import numpy as np
import torch
from dm_env import specs
import wandb

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle

from agent.potil import POTILAgent

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec.shape
	cfg.action_shape = action_spec.shape
	cfg.suite_name="metaworld"

	# ugly hacks
	dict_cfg = dict(cfg)
	del dict_cfg['_target_']

	return POTILAgent(**dict_cfg)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.train_env.observation_spec(),
								self.train_env.action_spec(), cfg.agent)

		# self.expert_replay_loader = make_expert_replay_loader(
		# 	self.cfg.expert_dataset, self.cfg.batch_size // 2, self.cfg.num_demos, self.cfg.obs_type)
		# self.expert_replay_iter = iter(self.expert_replay_loader)

		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

		self.expert_data_path = f"{rewardlearning_vid_repo_root}/ROT/ROT/expert_demos/{self.cfg.task_name}/expert_data.hdf"
		self.expert_data_ptr = H5PyTrajDset(self.expert_data_path, read_only_if_exists=True)
		self.expert_data = [d for d in self.expert_data_ptr][:self.cfg.num_demos]

		# with open(self.cfg.expert_dataset, 'rb') as f:
		# 	if self.cfg.obs_type == 'pixels':
		# 		self.expert_demo, _, _, self.expert_reward = pickle.load(f)
		# 	elif self.cfg.obs_type == 'features':
		# 		_, self.expert_demo, _, self.expert_reward = pickle.load(f)

		# expert demo is just 125 x 9 x 84 x 84, so should be hot swappable
		# we now have num_trajs x traj_length x 3 x 84 x 84
		self.expert_demo = np.concatenate([np.expand_dims(np.array(d[0]), axis=0) for d in self.expert_data])
		self.expert_reward = np.mean([np.sum(d[2]) for d in self.expert_data])

		project_name = "rewardlearningvid-metaworld-rot"
		date_str = datetime.today().strftime('%Y-%m-%d')
		folder_substr = "rot_image"
		exp_str = f"{date_str}/{folder_substr}/{self.cfg.task_name}-{folder_substr}-seed-{cfg.seed}"
		wandb_mode = "online"
		wandb.init(project=project_name, name=exp_str, mode=wandb_mode, dir=self.work_dir, settings=wandb.Settings(start_method='fork'))
		print(f"wandb is using {wandb.run.name} in {wandb_mode} mode")

	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		# self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		# self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.train_env = ImageMetaworldEnv(self.cfg.task_name, camera_name="left_cap2", high_res_env=False)
		self.eval_env = ImageMetaworldEnv(self.cfg.task_name, camera_name="left_cap2", high_res_env=False)

		# create replay buffer
		data_specs = [
			self.train_env.observation_spec(),
			self.train_env.action_spec(),
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount')
		]

		self.replay_storage = ReplayBufferStorage(data_specs,
												  self.work_dir / 'buffer')

		self.replay_loader = make_replay_loader(
			self.work_dir / 'buffer', self.cfg.replay_buffer_size,
			self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
			self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount)

		self._replay_iter = None
		self.expert_replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat

	@property
	def replay_iter(self):
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		step, episode, total_reward = 0, 0, 0
		eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

		if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
			paths = []

		og_total_rewards = []
		total_rewards = []
		successes = []
		succeededs = []
		while eval_until_episode(episode):
			if self.cfg.suite.name == 'metaworld':
				path = []
			time_step = self.eval_env.reset()
			self.video_recorder.init(self.eval_env, enabled=(episode == 0))

			succeeded = False
			per_episode_og_reward = 0
			per_episode_reward = 0

			while not time_step.last():
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(time_step.observation,
											self.global_step,
											eval_mode=True)
				time_step = self.eval_env.step(action)
				path.append(time_step.success)
				self.video_recorder.record(self.eval_env)
				total_reward += time_step.reward
				step += 1

				per_episode_reward += time_step.reward
				per_episode_og_reward += self.eval_env.get_last_received_reward()
				succeeded |= int(time_step.success)

			total_rewards.append(per_episode_reward)
			og_total_rewards.append(per_episode_og_reward)

			successes.append(time_step.success)
			succeededs.append(succeeded)

			episode += 1
			self.video_recorder.save(f'{self.global_frame}.mp4')
			paths.append(1 if np.sum(path)>10 else 0)

		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
			if repr(self.agent) != 'drqv2':
				log('expert_reward', self.expert_reward)
			if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
				log("success_percentage", np.mean(paths))
		metrics_dict = {
            'eval/episode_reward': np.mean(total_rewards),
            'eval/og_episode_reward': np.mean(og_total_rewards),
            'eval/success_rate': np.mean(successes),
            'eval/succeeded_rate': np.mean(succeededs),
			"eval/episode_length": step * self.cfg.suite.action_repeat / episode,
			"eval/episode": self.global_episode,
			"eval/step": self.global_step,
			"eval/expert_reward": self.expert_reward,
			"eval/success_percentage": np.mean(paths),
		}
		wandb.log(metrics_dict, step=self.global_step)


	def train_il(self):
		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0
		og_episode_reward = 0

		time_steps = list()
		observations = list()
		actions = list()

		time_step = self.train_env.reset()
		time_steps.append(time_step)
		observations.append(time_step.observation)
		actions.append(time_step.action)

		if repr(self.agent) == 'potil':
			if self.agent.auto_rew_scale:
				self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

		self.train_video_recorder.init(time_step.observation)
		metrics = None
		while train_until_step(self.global_step):
			if time_step.last():
				self._global_episode += 1
				if self._global_episode % 10 == 0:
					self.train_video_recorder.save(f'{self.global_frame}.mp4')
				# wait until all the metrics schema is populated
				observations = np.stack(observations, 0)
				actions = np.stack(actions, 0)
				if repr(self.agent) == 'potil':
					new_rewards = self.agent.ot_rewarder(
						observations, self.expert_demo, self.global_step)
					new_rewards_sum = np.sum(new_rewards)
				elif repr(self.agent) == 'dac':
					new_rewards = self.agent.dac_rewarder(observations, actions)
					new_rewards_sum = np.sum(new_rewards)

				if repr(self.agent) == 'potil':
					if self.agent.auto_rew_scale:
						if self._global_episode == 1:
							self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
								np.abs(new_rewards_sum))
							new_rewards = self.agent.ot_rewarder(
								observations, self.expert_demo, self.global_step)
							new_rewards_sum = np.sum(new_rewards)

				for i, elt in enumerate(time_steps):
					elt = elt._replace(
						observation=time_steps[i].observation)
					if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
							elt = elt._replace(reward=new_rewards[i])
					self.replay_storage.add(elt)

				if metrics is not None:
					# log stats
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.suite.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame,
													  ty='train') as log:
						log('fps', episode_frame / elapsed_time)
						log('total_time', total_time)
						log('episode_reward', episode_reward)
						log('episode_length', episode_frame)
						log('episode', self.global_episode)
						log('buffer_size', len(self.replay_storage))
						log('step', self.global_step)
						if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
								log('expert_reward', self.expert_reward)
								log('imitation_reward', new_rewards_sum)
					metrics_dict = {
						"train/fps": episode_frame / elapsed_time,
						"train/total_time": total_time,
						"train/episode_reward": episode_reward,
                        'train/og_episode_reward': og_episode_reward,
						"train/episode_length": episode_frame,
						"train/episode": self.global_episode,
						"train/buffer_size": len(self.replay_storage),
						"train/step": self.global_step,
						"train/expert_reward": self.expert_reward,
						"train/imitation_reward": new_rewards_sum,
					}
					wandb.log(metrics_dict, step=self.global_step)

				# reset env
				time_steps = list()
				observations = list()
				actions = list()

				time_step = self.train_env.reset()
				time_steps.append(time_step)
				observations.append(time_step.observation)
				actions.append(time_step.action)
				self.train_video_recorder.init(time_step.observation)
				# try to save snapshot
				if self.cfg.suite.save_snapshot:
					self.save_snapshot()
				episode_step = 0
				episode_reward = 0
				og_episode_reward = 0

			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(),
								self.global_frame)
				self.eval()

			# sample action
			with torch.no_grad(), utils.eval_mode(self.agent):
				action = self.agent.act(time_step.observation,
										self.global_step,
										eval_mode=False)

			# try to update the agent
			if not seed_until_step(self.global_step):
				# Update
				metrics = self.agent.update(self.replay_iter, self.expert_replay_iter,
											self.global_step, self.cfg.bc_regularize)
				self.logger.log_metrics(metrics, self.global_frame, ty='train')
				wandb.log(metrics, step=self.global_step)

			# take env step
			time_step = self.train_env.step(action)
			episode_reward += time_step.reward
			og_episode_reward += self.train_env.get_last_received_reward()

			time_steps.append(time_step)
			observations.append(time_step.observation)
			actions.append(time_step.action)

			self.train_video_recorder.record(time_step.observation)
			episode_step += 1
			self._global_step += 1

	def save_snapshot(self):
		snapshot = self.work_dir / 'snapshot.pt'
		keys_to_save = ['timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
	from train import WorkspaceIL as W
	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load weights
	if cfg.load_bc:
		snapshot = Path(cfg.bc_weight)
		if snapshot.exists():
			print(f'resuming bc: {snapshot}')
			workspace.load_snapshot(snapshot)

	workspace.train_il()


if __name__ == '__main__':
	main()
