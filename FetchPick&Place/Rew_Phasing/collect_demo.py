import os
import argparse
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo
from pick_and_fetch import PF

def run(args):
    env = PF()#make_env(args.env_id)

    '''algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )'''
    
    algo = env.goToGoal

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        'size{}_std{}_prand{}.pth'.format(args.buffer_size, args.std, args.p_rand)
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=False)
    p.add_argument('--env_id', type=str, default='PF')
    p.add_argument('--buffer_size', type=int, default=1024*75)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
