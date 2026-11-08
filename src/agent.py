import torch as th
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import time
from contextlib import nullcontext
from torchvision.models import resnet18
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pdb

from model.cnn import MyCNN, MyCNNCoarse
from model.actor import Actor
from model.critic import Critic


def resolve_return_gamma(args):
    """Resolve the discount factor used for PPO cumulative returns.

    Step-granularity rewards are discounted by ``reward_discount_gamma`` when
    the config opts into it, and fall back to ``gamma`` otherwise so existing
    configs keep their historical behavior.  Global (terminal whole-placement)
    rewards are not discounted and use ``global_reward_gamma``.

    ``gamma`` is deliberately not used for the step discount here because it
    also drives PLM community resolution in ``problem_instance.py``; changing
    it for the discount would silently change the state representation.
    """
    return_gamma = getattr(args, 'reward_discount_gamma', args.gamma)
    if getattr(args, 'reward_granularity', 'step') == 'global':
        return_gamma = getattr(args, 'global_reward_gamma', 1.0)
    return return_gamma


class PPOAgent():
    def __init__(self, args, inference_only=False):
        self.args = args
        self.inference_only = inference_only
        # Worker rollouts overwrite every actor parameter with the driver's
        # state, so they do not need to load ImageNet weights or build a critic.
        self.resnet = resnet18(pretrained=not inference_only)
        actor_arch = getattr(args, 'actor_arch', 'parallel')
        if actor_arch == 'sequential':
            k = getattr(args, 'local_summary_k', 4)
            canvas_num = 2 if getattr(args, 'prototype_flag', False) else 1
            self.cnn = MyCNN(args=args, out_channels=k).to(args.device)
            self.cnn_coarse = MyCNNCoarse(args=args, res_net=self.resnet, input_dim=canvas_num + k).to(args.device)
        else:
            self.cnn = MyCNN(args=args).to(args.device)
            self.cnn_coarse = MyCNNCoarse(args=args, res_net=self.resnet).to(args.device)

        self.actor = Actor(args=args, cnn=self.cnn, cnn_coarse=self.cnn_coarse).float().to(args.device)
        self.critic = None if inference_only else Critic(args=args, cnn_coarse=self.cnn_coarse).float().to(args.device)

        self.actor_opt = None if inference_only else optim.Adam(self.actor.parameters(), args.lr)
        self.critic_opt = None if inference_only else optim.Adam(self.critic.parameters(), args.lr)

        self.buffer = []
        self.batch_size = args.batch_size
        self.buffer_capacity = 0 if inference_only else args.buffer_size * args.n_macro
        self.counter = 0
        self.training_step = 0

        self.training = False
        self.state_rebuilder = None
        self.mixed_precision = bool(getattr(args, 'mixed_precision', False)) and self.args.device != 'cpu'
        configured_amp_dtype = getattr(th, getattr(args, 'amp_dtype', 'float16'), th.float16)
        # Some PyTorch CPU builds validate autocast dtype even when disabled;
        # use float32 whenever AMP is inactive to keep the CPU path portable.
        self.amp_dtype = configured_amp_dtype if self.mixed_precision else th.float32

    def set_state_rebuilder(self, rebuilder):
        """Set the environment callback used to materialize compact states."""
        self.state_rebuilder = rebuilder

    def _build_state_batch(self):
        """Materialize rollout observations without changing baseline dtype."""
        if isinstance(self.buffer[0].state, dict):
            if self.state_rebuilder is None:
                raise RuntimeError("compact transitions require a state rebuilder")
            states = [self.state_rebuilder(t.state) for t in self.buffer]
            return th.from_numpy(np.concatenate(states, axis=0)).float()

        state_dtype = np.float16 if getattr(self.args, 'compact_transition', True) else np.float32
        return th.from_numpy(np.concatenate(
            [np.asarray(t.state, dtype=state_dtype) for t in self.buffer], axis=0
        )).float()
    
    def select_action(self, state:np.array):
        """
        Select actions for a batch of environments.

        Args:
            state (np.array): Batched state input, shape [batch_size, ...].

        Returns:
            actions (list): List of selected actions for each environment, length=batch_size.
            action_log_probs (list): List of log probabilities for each selected action, length=batch_size.

        This method converts the input numpy array to a torch tensor, computes action probabilities
        using the actor network, samples or selects the most probable action for each environment
        (depending on training/eval mode), and returns the results as Python lists.
        """
        # state: [batch, ...], do not add unsqueeze(0)
        state = th.from_numpy(state).float().to(self.args.device)
        # Rollout inference never needs autograd metadata.  inference_mode
        # also disables version-counter/view tracking used by no_grad.
        with th.inference_mode():
            with (th.autocast(device_type=state.device.type, dtype=self.amp_dtype)
                  if self.mixed_precision else nullcontext()):
                action_prob = self.actor(state)  # [batch, N]
        # Autocast is limited to CNN/MLP kernels.  Keep probability
        # construction and sampling in FP32 so low-precision logits cannot
        # trigger distribution kernels or alter the non-AMP baseline path.
        dist = Categorical(action_prob.float())
        if self.training:
            action = dist.sample()  # [batch]
        else:
            action = dist.probs.argmax(dim=1)  # [batch]
        action_log_prob = dist.log_prob(action)  # [batch]
        return action.cpu().numpy().tolist(), action_log_prob.cpu().numpy().tolist()
    
    def update(self):
        if self.counter % self.buffer_capacity == 0:
            t_start = time.time() #########
            stage_batch = getattr(self.args, 'batch_device_copy', True)
            # Compact snapshots are rebuilt on demand.  For the explicit
            # non-compact baseline, preserve the original FP32 host buffer;
            # otherwise this path would silently change the baseline.
            state = self._build_state_batch()
            action = th.tensor(np.array([t.action for t in self.buffer]), dtype=th.float).view(-1, 1)
            reward = th.stack([t.reward for t in self.buffer]).view(-1, 1)
            # reward = th.tensor(np.array([t.reward for t in self.buffer]), dtype=th.float).view(-1, 1).to(self.args.device)
            old_action_log_prob = th.tensor(np.array([t.action_log_prob for t in self.buffer]), dtype=th.float).view(-1, 1)
            done = th.tensor(np.array([t.done for t in self.buffer], dtype=np.int32), dtype=th.int32).view(-1, 1)
            if stage_batch:
                state = state.to(self.args.device)
                action = action.to(self.args.device)
                reward = reward.to(self.args.device)
                old_action_log_prob = old_action_log_prob.to(self.args.device)
                done = done.to(self.args.device)
            del self.buffer[:]
            
            # Calculate cumulative returns
            target_list = []
            target = 0
            return_gamma = resolve_return_gamma(self.args)
            for i in range(reward.shape[0]-1, -1, -1):
                if done[i, 0] == 1:
                    target = 0
                r = reward[i, 0].item()
                target = r + return_gamma * target
                target_list.append(target)
            target_list.reverse()
            target_v_all = th.tensor(np.array([t for t in target_list]), dtype=th.float32).view(-1, 1)
            if stage_batch:
                target_v_all = target_v_all.to(self.args.device)
            
            actor_loss_lst  = []
            critic_loss_lst = []
            batch_size = min(self.batch_size, self.buffer_capacity)
            for _ in range(self.args.epoch):
                for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), batch_size, False),
                    disable = self.args.disable_tqdm):
                    self.training_step += 1
                    batch_state = state[index]
                    if not stage_batch:
                        batch_state = batch_state.to(self.args.device)
                        batch_action = action[index].to(self.args.device)
                        batch_old_action_log_prob = old_action_log_prob[index].to(self.args.device)
                        batch_target_v = target_v_all[index].to(self.args.device)
                    else:
                        batch_action = action[index]
                        batch_old_action_log_prob = old_action_log_prob[index]
                        batch_target_v = target_v_all[index]
                    with (th.autocast(device_type=batch_state.device.type, dtype=self.amp_dtype)
                          if self.mixed_precision else nullcontext()):
                        action_prob = self.actor(batch_state)
                    dist = Categorical(action_prob.float())
                    action_log_prob = dist.log_prob(batch_action.squeeze())
                    ratio = th.exp(action_log_prob - batch_old_action_log_prob.squeeze())
                    target_v = batch_target_v
                    # The critic currently uses only place_idx (column 0). Avoid
                    # copying the full multi-channel placement state twice.
                    with (th.autocast(device_type=batch_state.device.type, dtype=self.amp_dtype)
                          if self.mixed_precision else nullcontext()):
                        critic_output = self.critic(batch_state[:, :1])
                    # Some CUDA builds have no BF16 smooth_l1 backward
                    # kernel.  Compute the value loss and advantage in FP32
                    # for every opt-in AMP dtype; with AMP disabled these
                    # references are unchanged and preserve the baseline.
                    if self.mixed_precision:
                        critic_loss_output = critic_output.float()
                        critic_loss_target = target_v.float()
                    else:
                        critic_loss_output = critic_output
                        critic_loss_target = target_v
                    advantage = (critic_loss_target - critic_loss_output).detach()
                    L1 = ratio * advantage.squeeze() 
                    L2 = th.clamp(ratio, 1-self.args.clip_param, 1+self.args.clip_param) * advantage.squeeze() 
                    action_loss = -th.min(L1, L2).mean() 

                    self.actor_opt.zero_grad()
                    action_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                    self.actor_opt.step()

                    value_loss = nn.functional.smooth_l1_loss(
                        critic_loss_output, critic_loss_target
                    )
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                    self.critic_opt.step()

                    actor_loss_lst.append(action_loss.cpu().item())
                    critic_loss_lst.append(value_loss.cpu().item())

            t_end = time.time() #########
            print("Training time: {:.2f}s".format(t_end - t_start)) #########
            return t_end - t_start, np.mean(actor_loss_lst), np.mean(critic_loss_lst)   
        else:
            return 0, -1, -1

    def store_transition(self, transition):
        self.counter += 1
        self.buffer.append(transition)

    def store_rollout(self, rollout):
        """Append a worker-produced rollout without per-step Ray calls."""
        for transition in rollout:
            self.store_transition(transition)

    def save_model(self, episode, best_so_far=False, path=None, filename=None):
        base_path = path if path is not None else self.args.log_dir
        if filename is not None:
            file_name = filename
        else:
            if best_so_far:
                file_name = f'best_model_ep{episode}.pt'
            else:
                file_name = f'model_ep{episode}.pt'
        os.makedirs(base_path, exist_ok=True)
        target_path = os.path.join(base_path, file_name)
        th.save(
            {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'episode': episode,
                'reward_weights': list(self.args.trade_off_coeff),
            },
            target_path
        )
    
    def load_model(self, checkpoint_path):
        checkpoint = th.load(checkpoint_path, map_location=th.device(self.args.device))
        if 'actor' in checkpoint and 'critic' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            critic_state = checkpoint['critic']
        else:
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            critic_state = checkpoint.get('critic_state_dict')
        if self.critic is not None and critic_state is not None:
            self.critic.load_state_dict(critic_state)
        if 'reward_weights' in checkpoint:
            self.set_reward_weights(checkpoint['reward_weights'])

    def set_reward_weights(self, weights):
        """Keep checkpoint metadata and the actor soft mask in sync."""
        self.args.trade_off_coeff = list(weights)
        self.actor.state_parsing.set_reward_weights(weights)
    
    def train(self):
        self.training = True
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        if self.critic is not None:
            self.critic.eval()
