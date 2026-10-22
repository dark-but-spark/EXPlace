import numpy as np
import torch as th
import pdb


def get_batch_state_from_buffer(buffer, index):
    buffer_batch = [buffer[i] for i in index]

    # Find the maximum number of corners in the batch
    max_corners = max(len(t.state['corners']) for t in buffer_batch)
    
    # Initialize lists to store padded corners and padding masks
    padded_corners = []
    corner_padding_masks = []
    
    for t in buffer_batch:
        corners = t.state['corners']
        current_corners = len(corners)
        
        # Create padding mask: 1 for real corners, 0 for padded corners
        padding_mask = [1] * current_corners + [0] * (max_corners - current_corners)
        corner_padding_masks.append(padding_mask)
        
        # Pad corners with zeros
        if current_corners < max_corners:
            
            # Create zero padding
            padding_size = max_corners - current_corners
            zero_padding = np.zeros((padding_size, corners.shape[1]))
            
            # Concatenate original corners with zero padding
            padded_corner = np.concatenate([corners, zero_padding], axis=0)
            padded_corners.append(padded_corner)
        else:
            # No padding needed
            padded_corners.append(corners)
    
    return {
        'corners': th.tensor(padded_corners),
        'nodes': th.tensor([t.state['nodes'] for t in buffer_batch]),
        'idx': th.tensor([t.state['idx'] for t in buffer_batch]),
        'prototype': th.tensor([t.state['prototype'] for t in buffer_batch]),
        'corner_mask': th.tensor(corner_padding_masks, dtype=th.bool)  # True for real corners, False for padded
    }

class StateParsing:
    def __init__(self, args) -> None:
        self.args = args
        self.grid = args.grid
        self.g2 = self.grid * self.grid
        self._reward_coeff_cache = {}
        self.set_reward_weights(args.trade_off_coeff)

    def set_reward_weights(self, weights):
        """Update the expert-mask weights used by the actor's soft mask."""
        expected = len(self.args.used_masks) - 1
        if len(weights) != expected:
            raise ValueError(f"expected {expected} reward weights, got {len(weights)}")
        self.reward_weights = np.asarray(weights, dtype=np.float64).copy()
        # Reward weights can change between physical-evaluation updates.  Any
        # device/dtype copies made for the actor decoder must then be rebuilt.
        self._reward_coeff_cache.clear()

    def _reward_coeff_tensor(self, state):
        dtype = state.dtype if getattr(self.args, 'policy_mask_fp32', False) else th.float64
        key = (state.device.type, state.device.index, dtype)
        coeff = self._reward_coeff_cache.get(key)
        if coeff is None:
            values = np.concatenate((self.reward_weights, np.array([10.0])))
            coeff = th.as_tensor(values, dtype=dtype, device=state.device)
            self._reward_coeff_cache[key] = coeff
        return coeff
    
    def get_state(self,
        place_idx, 
        canvas,
        masks,
        next_masks,
        size_x,
        size_y,
        prototype_canvas=None,
        ):
        state_parts = [np.array([place_idx]), canvas.reshape(-1)]
        if prototype_canvas is not None:
            state_parts.append(prototype_canvas.reshape(-1))
        state_parts.extend(mask.reshape(-1) for mask in masks.values())
        state_parts.extend(mask.reshape(-1) for mask in next_masks.values())
        state_parts.append(np.array([size_x / self.grid, size_y / self.grid]))

        # The policy immediately consumes float32 tensors. Constructing the
        # state in that dtype halves Ray traffic and PPO-buffer memory while a
        # single concatenate avoids repeatedly copying the growing state.
        state = np.concatenate(state_parts, dtype=np.float32)
        return state[np.newaxis, :]

    def compact_state(self, state):
        """Return the scalar placement metadata used by the value function.

        Actor transitions still use the full observation when sampled, but this
        helper provides a stable compact representation for buffers and tooling
        that only need the macro index and size metadata.
        """
        state = np.asarray(state)
        return np.asarray(state[..., :1], dtype=np.float32).copy()

    def state2canvas(self, state):
        """Extract canvas (and optional prototype_canvas) for sequential global input. Returns [B, canvas_num, grid, grid]."""
        canvas_num = 2 if self.args.prototype_flag else 1
        if len(state.shape) == 1:
            return state[1 : 1 + canvas_num * self.g2].reshape(canvas_num, self.grid, self.grid)
        elif len(state.shape) == 2:
            return state[:, 1 : 1 + canvas_num * self.g2].reshape(-1, canvas_num, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2input_global(self, state):
        # contains canvas, prototype_canvas, masks without position mask
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        if len(state.shape) == 1:
            state_1 = state[1 : 1 + self.g2 * (mask_num - 1 + canvas_num)].reshape(mask_num - 1 + canvas_num, self.grid, self.grid)
            state_2 = state[1 + self.g2 * (mask_num + canvas_num) : 1 + self.g2 * (mask_num * 2 - 1 + canvas_num)].reshape(mask_num - 1, self.grid, self.grid)
            return th.cat((state_1, state_2), dim=0)
        elif len(state.shape) == 2:
            state_1 = state[:, 1 : 1 + self.g2 * (mask_num - 1 + canvas_num)].reshape(-1, mask_num - 1 + canvas_num, self.grid, self.grid)
            state_2 = state[:, 1 + self.g2 * (mask_num + canvas_num) : 1 + self.g2 * (mask_num * 2 - 1 + canvas_num)].reshape(-1, mask_num - 1, self.grid, self.grid)
            return th.cat((state_1, state_2), dim=1)
        else:
            raise NotImplementedError

    def state2input_local(self, state):
        # contains all masks
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        if len(state.shape) == 1:
            return state[1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num * 2 + canvas_num)].reshape(mask_num * 2, self.grid, self.grid)
        elif len(state.shape) == 2:
            return state[:, 1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num * 2 + canvas_num)].reshape(-1, mask_num * 2, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2position_mask(self, state):
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        if len(state.shape) == 1:
            return state[1 + self.g2 * (mask_num - 1 + canvas_num) : 1 + self.g2 * (mask_num + 2)].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            return state[:, 1 + self.g2 * (mask_num - 1 + canvas_num) : 1 + self.g2 * (mask_num + canvas_num)].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError
    
    def state2soft_mask(self, state):
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        # The historical decoder used float64 coefficients.  Keep that
        # default for baseline runs, but let GPU/AMP experiments use the
        # policy input dtype and broadcast one coefficient per mask instead
        # of materializing a [batch, mask, grid^2] repeat tensor.
        trade_off_coeff = self._reward_coeff_tensor(state)
        if len(state.shape) == 1:
            trade_off_coeff = trade_off_coeff.view(mask_num, 1)
            masks = state[1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num + canvas_num)].reshape(mask_num, self.g2)
            soft_mask = th.sum(masks * trade_off_coeff, dim=0)
            return soft_mask
        elif len(state.shape) == 2:
            trade_off_coeff = trade_off_coeff.view(1, mask_num, 1)
            masks = state[:, 1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num + canvas_num)].reshape(-1, mask_num, self.g2)
            soft_mask = th.sum(masks * trade_off_coeff, dim=1)
            return soft_mask
        else:
            raise NotImplementedError
