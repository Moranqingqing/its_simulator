import logging
import os
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from wolf.examples.ddpg import DDPG

logger = logging.getLogger('ddpg')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SafeLayerDDPG(DDPG):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, checkpoint_dir=None, device='cuda'):
        super().__init__(gamma, tau, hidden_size, num_inputs, action_space, checkpoint_dir=checkpoint_dir)

        self.device = device

        self.safe_layer = nn.Sequential(
            nn.Linear(num_inputs + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

        self.safe_layer_optimizer = Adam(self.safe_layer.parameters(), lr=1e-3)

    def calc_action(self, state, action_noise=None, requires_grad=False):
        """
        Evaluates the action to perform in a given state

        Arguments:
            state:          State to perform the action on in the env. 
                            Used to evaluate the action.
            action_noise:   If not None, the noise to apply on the evaluated action
        """
        x = state.to(self.device)

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        with torch.set_grad_enabled(requires_grad):
            mu = self.actor(x)
            self.actor.train()  # Sets the actor in training mode
            mu = mu.data
            # During training we add noise for exploration
            if action_noise is not None:
                noise = torch.Tensor(action_noise.noise()).to(self.device)
                mu += noise

            # Clip the output according to the action space of the env
            # mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])
            mu = mu.clamp(-3, 3)
            
            inputs_to_safe_layer = torch.cat((x, mu), 1)
            output = self.safe_layer(inputs_to_safe_layer)
            output = 3*torch.tanh(output) # Range of tanh: (-1, 1)
        
        return output

    def update_params(self, batch, weight=0.1):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        with torch.no_grad():
            next_state_action_batch = torch.cat((next_state_batch, next_action_batch), 1)
            next_safe_action_batch = self.safe_layer(next_state_action_batch)
            next_safe_action_batch = 3 * torch.tanh(next_safe_action_batch)
            
        next_state_action_values = self.critic_target(next_state_batch, next_safe_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values
        
        with torch.no_grad():
            mu_action_batch = self.actor(state_batch)
        

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # TODO: Change the action_batch to the safe layer output? (I think so since reward is returned by taking the real action)
        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the safe layer
        self.safe_layer_optimizer.zero_grad()
        safe_action_batch = self.safe_layer(torch.cat((state_batch, mu_action_batch), 1))

        safe_action_batch = 3*torch.tanh(safe_action_batch)

        batch_size = state_batch.shape[0]
        veh_speed_batch = state_batch.gather(1, torch.tensor([0 for i in range(batch_size)], device=self.device).view(-1, 1))
        rel_speed_batch = -state_batch.gather(1, torch.tensor([1 for i in range(batch_size)], device=self.device).view(-1, 1))
        distance_headway_batch = state_batch.gather(1, torch.tensor([2 for i in range(batch_size)], device=self.device).view(-1, 1))
        veh_speed_limit_batch = state_batch.gather(1, torch.tensor([3 for i in range(batch_size)], device=self.device).view(-1, 1))
        
        # ttc = distance_headway_batch / rel_speed_batch
        # ttc = torch.where((ttc>=0.) & (ttc<=4.), ttc, torch.tensor(4., device=self.device))
        # collision_loss = nn.SmoothL1Loss()(ttc, torch.tensor([4 for i in range(batch_size)], device=self.device).view(-1, 1))

        ##FIXME: 0.1 might need to move to some variable
        predicted_veh_speed_batch = veh_speed_batch + 0.1 * safe_action_batch # timesteps is 1 sec 
        speed_loss = nn.SmoothL1Loss()(predicted_veh_speed_batch, veh_speed_limit_batch)

        mse_loss = nn.SmoothL1Loss()(action_batch, safe_action_batch)

        constrain_loss = mse_loss + weight * speed_loss
        constrain_loss.backward()
        self.safe_layer_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep, replay_buffer, eval_flag=False):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'

        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        eval_str = 'eval_' if eval_flag else ''
        checkpoint_name = self.checkpoint_dir + f'/{eval_str}ep_{last_timestep}.pth.tar'
        logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            "safe_layer": self.safe_layer.state_dict()
        }
        logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def load_checkpoint(self, checkpoint_path=None, training=False):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.

        Arguments:
            checkpoint_path:    File to load the model from

        """

        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint...({})".format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.safe_layer.load_state_dict(checkpoint['safe_layer'])

            replay_buffer = None
            # if training:
            #     replay_buffer = checkpoint['replay_buffer']
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')