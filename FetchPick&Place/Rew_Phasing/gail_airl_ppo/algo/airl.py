import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import AIRLDiscrim
import os
import numpy as np

class AIRL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.98, rollout_length=10000, mix_buffer=1,
                 batch_size=1024, lr_actor=7e-5, lr_critic=7e-5, lr_disc=7e-5,
                 units_actor=(256,256,256), units_critic=(256,256,256),
                 units_disc_r=(128, 128), units_disc_v=(128, 128),
                 epoch_ppo=4, epoch_disc=4, clip_eps=0.2, lambd=0.95,
                 coef_ent=0.001, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.rew_frac = 0.15#0.55#1.0
        self.episodes = 0

    def update(self):
        self.learning_steps += 1
        if(self.rew_frac > 0.0):
            for _ in range(self.epoch_disc):
                self.learning_steps_disc += 1

                # Samples from current policy's trajectories.
                states, _, _, dones, log_pis, next_states = \
                    self.buffer.sample(self.batch_size)
                # Samples from expert's demonstrations.
                states_exp, actions_exp, _, dones_exp, next_states_exp = \
                    self.buffer_exp.sample(self.batch_size)
                # Calculate log probabilities of expert actions.
                with torch.no_grad():
                    log_pis_exp = self.actor.evaluate_log_pi(
                        states_exp, actions_exp)
                # Update discriminator.
                self.update_disc(
                    states, dones, log_pis, next_states, states_exp,
                    dones_exp, log_pis_exp, next_states_exp
                )

        if((self.episodes+1)%200 == 0):
            if(self.rew_frac > 0.25):
                self.rew_frac= max(self.rew_frac-0.015, 0.0)
            else:
                self.rew_frac= max(self.rew_frac-0.01, 0.0)
        self.episodes += 1
        print("Rew frac=", self.rew_frac)

        # We don't use reward signals here,
        states, actions, r, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        if(self.rew_frac > 0.0):
            rewards = r+self.rew_frac*self.disc.calculate_reward(
                states, dones, log_pis, next_states)
        else:
            rewards = r
            
        f = open("rewards.txt", "a+")
        f.write(str(np.mean(rewards.numpy()))+","+str(np.mean(r.numpy()))+","+str(self.rew_frac)+"\n")
        f.close()
        print(str(np.mean(rewards.numpy()))+","+str(np.mean(r.numpy())))

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            print('loss/disc', loss_disc.item(), self.learning_steps)
            '''writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)'''

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            print('stats/acc_pi', acc_pi, self.learning_steps)
            print('stats/acc_exp', acc_exp, self.learning_steps)
            '''writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)'''
            
    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(), 'actor.pth'
        )
        torch.save(
            self.critic.state_dict(), 'critic.pth'
        )
        self.disc.save_models(save_dir)
