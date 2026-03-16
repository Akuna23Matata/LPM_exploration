
import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from exploration import algo, utils
from exploration.arguments import get_args
from exploration.envs import make_vec_envs
from exploration.model import Policy
from exploration.storage import RolloutStorage

from exploration.models import *
from stable_baselines3.common.running_mean_std import RunningMeanStd
import exploration.environments
from exploration.algo.random import RandomAgent, RandomPolicy

import wandb

def main():
    # Setup
    args = get_args()
    # warnings
    if args.use_bn:
        print("Using BatchNorm in the model")
    if args.use_ln:
        print("Using LayerNorm in the model")
    #
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    # Initialize wandb
    wandb.init(
        project="mr_exp",
        name=f"{args.algo}_{args.env_name}_seed{args.seed}",
        config=vars(args),
        dir=log_dir,
        sync_tensorboard=False,
        monitor_gym=False
    )

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Environment and Policy 
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    train_model = False
    is_vision = len(envs.observation_space.shape) > 1

    if args.algo == 'random':
        actor_critic = RandomPolicy(envs, args.num_processes)
    else:
        actor_critic = Policy(
            envs.observation_space,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)

    # Algorithm
    if args.algo in ['ppo-rnd', 'ppo-lpm', 'ppo-ngu']:
        obs_dim = envs.observation_space.shape
        discrete_action = envs.action_space.__class__.__name__ == "Discrete"
        if envs.action_space.__class__.__name__ == "Discrete":
            act_dim = envs.action_space.n
        elif envs.action_space.__class__.__name__ == "Box":
            act_dim = envs.action_space.shape[0]
        elif envs.action_space.__class__.__name__ == "MultiBinary":
            act_dim = envs.action_space.shape[0]
        else:
            raise NotImplementedError

        intr_ret_rms = RunningMeanStd()
        intr_ret = np.zeros((args.num_processes, 1))
        
        if args.use_dones:
            ext_coeff = 1. # 1 # Sparse tasks: 1000
            int_coeff = 1e-5  # 0.001 or 0.01
        else:
            ext_coeff = 0. # 1 # Sparse tasks: 1000
            int_coeff = 1.  # 0.001 or 0.01

        train_model = True

        if is_vision:
            hidden_dim = 512
            state_dim = 512
        else:
            hidden_dim = 32 
            state_dim = obs_dim

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo-rnd':
        model = RND(
            obs_dim,
            act_dim,
            state_dim,
            hidden_dim,
            device,
            use_bn=args.use_bn,
            use_ln=args.use_ln
        )
    elif args.algo == 'ppo-lpm':
        model = LPM(
            obs_dim,
            act_dim,
            state_dim,
            hidden_dim,
            device,
            use_bn=args.use_bn,
            use_ln=args.use_ln,
            buffer_size=args.lpm_buffer_size,
        )
    elif args.algo == 'ppo-ngu':
        model = NGU(
            obs_dim,
            act_dim,
            state_dim,
            hidden_dim,
            device,
            use_bn=args.use_bn,
            use_ln=args.use_ln,
            ngu_knn_k=args.ngu_knn_k,
            ngu_dst_momentum=args.ngu_dst_momentum,
            ngu_use_rnd=args.ngu_use_rnd,
            rnd_err_norm=args.ngu_rnd_err_norm,
        )
    elif args.algo == 'random':
        agent = RandomAgent()

    if train_model:
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        
        model.to(device)
        model_lr = args.lr
        
        # For LPM, we need separate optimizers for RND and error predictor
        if args.algo == 'ppo-lpm':
            # Optimizer for RND predictor network (inside LPM)
            model_optimizer = optim.Adam(model.rnd.predictor_network.parameters(), lr=model_lr)
            # Optimizer for error predictor network (now unified for both vector and visual)
            error_optimizer = optim.Adam(model.error_predictor.parameters(), lr=model_lr)
        elif args.algo == 'ppo-ngu':
            # For NGU, optimize embedding, inverse model, and RND predictor (not RND target)
            trainable_params = list(model.embedding_network.parameters()) + list(model.inverse_model.parameters())
            if model.ngu_use_rnd:
                trainable_params += list(model.rnd_predictor.parameters())
            model_optimizer = optim.Adam(trainable_params, lr=model_lr)
        else:
            model_optimizer = optim.Adam(model.parameters(), lr=model_lr)
        
        model.train()

    # Setup rollouts and Episode Queue
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs_rms = None
    if is_vision:

        init_env = make_vec_envs(args.env_name, args.seed, 1, args.gamma, None, 'cpu', False)

        # normalize obs
        init_env.reset()
        random_agent = RandomPolicy(envs, 1)
        print('Start to initialize observation normalization parameter.....')
        obs_init = []
        steps = 0
        pre_obs_norm_step = int(1e4)
        while steps < pre_obs_norm_step:
            steps += 1
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = random_agent.act(None, None, None)
            no, _, _, _ = init_env.step(action)
            
            obs_init.append(np.asarray(no.detach().cpu()).astype(np.float32))
        
        obs_init = np.array(obs_init)
        obs_mean = torch.Tensor(np.mean(obs_init, 0).astype(np.float32)).to(device)
        obs_std = torch.Tensor([np.std(obs_init, 0).mean().astype(np.float32)]).to(device)

        if args.algo in ['ppo-rnd', 'ppo-lpm', 'ppo-ngu']:
            obs_init = torch.from_numpy(obs_init).float().to(device)
            obs_init = ((obs_init - obs_mean) / (1e-8 + obs_std)).reshape(pre_obs_norm_step, *obs_dim)
            model.get_feature_moments(obs_init)
        
        # Initialize NGU episodic memory with correct number of environments
        if args.algo == 'ppo-ngu':
            model.init_episodic_memory(args.num_processes)
            print(f"NGU: Initialized episodic memory for {args.num_processes} parallel environments")
        
        del init_env

        print('End to initialize...')
        
        obs_rms = RunningMeanStd()
        obs_rms.mean = obs_mean
        obs_rms.var = obs_std**2
    else:
        obs_mean = 0 
        obs_std = 1 

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_return = -1e+8
    best_length = -1e+8

    if args.env_name in ['MagellanAnt-v2', 'MountainCarSparse-v0', 'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
        states_buffer = []
        update_state_buffer = []
        overall_blocks = []
    
    # Tracking for Montezuma's Revenge
    if 'Montezuma' in args.env_name:
        all_positions = set()  # Track all unique (x, y, room) positions
        all_rooms = set()  # Track all unique rooms visited
        update_positions = set()  # Positions in current update interval
        update_rooms = set()  # Rooms in current update interval
        episode_pos_counts = []  # Track positions per episode
        episode_room_counts = []  # Track rooms per episode

    # Training
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        start = time.time()
        update_episode_returns = []
        update_episode_lengths = []

        if args.use_linear_lr_decay and args.algo != 'random':
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    (rollouts.obs[step] - obs_mean) / (obs_std + 1e-8), rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # It causes problem otherwise
            if args.env_name == 'MarioBrosNoFrameskip-v4':
                next_obs, reward, done, infos = envs.step(action.to('cpu'))
            else:    
                # Obs reward and next obs
                next_obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    discounted_returns = info['episode']['r']
                    ep_length = info['episode']['l']
                    episode_rewards.append(discounted_returns)
                    episode_lengths.append(ep_length)
                    update_episode_returns.append(discounted_returns)
                    update_episode_lengths.append(ep_length)
                    if discounted_returns > best_return:
                        best_return = discounted_returns
                    if ep_length > best_length:
                        best_length = ep_length
                if args.env_name in ['MagellanAnt-v2', 'MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
                    states_buffer.append(info['obs'])
                    update_state_buffer.append(info['obs'])
                
                # Track Montezuma's Revenge coverage
                if 'Montezuma' in args.env_name:
                    if 'x' in info and 'y' in info and 'room' in info:
                        pos = (info['x'], info['y'], info['room'])
                        all_positions.add(pos)
                        update_positions.add(pos)
                        all_rooms.add(info['room'])
                        update_rooms.add(info['room'])
                    
                    # Track episode-level statistics
                    if 'mz_episode' in info:
                        episode_pos_counts.append(info['mz_episode']['pos_count'])
                        episode_room_counts.append(len(info['mz_episode']['visited_rooms']))
            
            if args.use_dones:
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
            else:
                masks = torch.FloatTensor([[1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[1.0] for info in infos])

            rollouts.insert(next_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            obs = next_obs

        # Normalize observations
        rollouts.obs = ( rollouts.obs - obs_mean) / (obs_std + 1e-8)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        
        if train_model:
            o = rollouts.obs[:-1].detach().reshape(-1, *obs_dim)
            no = rollouts.obs[1:].detach().reshape(-1, *obs_dim)
            ext_reward = rollouts.rewards.detach() 
                    
            if is_vision:
                a = rollouts.actions.detach().reshape(-1, 1)
                a = utils.cat_act_to_vector(a, act_dim, device)
            else:
                a = rollouts.actions.detach().reshape(-1, act_dim)

            max_size = 2048
            if obs.shape[0] > max_size:
                curiosities = []
                with torch.no_grad():
                    for indx in range(0, obs.shape[0], max_size):
                        if args.algo == 'ppo-ngu':
                            # NGU needs masks for episodic memory reset
                            masks_batch = rollouts.masks[:-1].detach().reshape(-1, 1)[indx:indx+max_size]
                            c = model.curiosity(o[indx:indx+max_size], a[indx:indx+max_size], no[indx:indx+max_size], masks=masks_batch)
                        else:
                            c = model.curiosity(o[indx:indx+max_size], a[indx:indx+max_size], no[indx:indx+max_size])
                        curiosities.append(c)
                curiosity = torch.cat(curiosities)
            else:
                with torch.no_grad():
                    if args.algo == 'ppo-ngu':
                        # NGU needs masks for episodic memory reset
                        masks_batch = rollouts.masks[:-1].detach().reshape(-1, 1)
                        curiosity = model.curiosity(o, a, no, masks=masks_batch)
                    else:
                        curiosity = model.curiosity(o, a, no)
            
            
            intr_rew = curiosity.reshape(ext_reward.shape).detach().cpu().numpy()
            intr_rew = np.clip(intr_rew, -3 * np.sqrt(intr_ret_rms.var), 3 * np.sqrt(intr_ret_rms.var))

            upd_intr_ret = []
            for idx in range(intr_rew.shape[0]):
                intr_ret = intr_ret * args.gamma + intr_rew[idx]
                upd_intr_ret.append(intr_ret)

            upd_intr_ret = np.reshape(np.stack(upd_intr_ret), [args.num_steps * args.num_processes, 1])
            mean, std, count = np.mean(upd_intr_ret), np.std(upd_intr_ret), len(upd_intr_ret)
            intr_ret_rms.update_from_moments(mean, std ** 2, count)

            intr_rew = intr_rew / np.sqrt(intr_ret_rms.var + 1e-8) 

            rollouts.rewards = ext_coeff * ext_reward + int_coeff * torch.Tensor(intr_rew).to(device) 

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        # model update
        if train_model:
            batch_size = args.num_steps * args.num_processes // args.num_mini_batch
            # if is_vision:
            for _ in range(args.ppo_epoch):
                data_generator = rollouts.model_generator(args.num_mini_batch)
                for sample in data_generator:
                    obs_batch, next_obs_batch, act_batch = sample
                    indices = None
                    if is_vision:
                        act_batch = utils.cat_act_to_vector(act_batch, act_dim, device)

                    model_optimizer.zero_grad()
                    loss = model.loss(obs_batch, act_batch, next_obs_batch, indices=indices, actor_critic=actor_critic).mean()
                    loss.backward()
                    model_optimizer.step()
                    
                    # For LPM, also train the error predictor
                    if args.algo == 'ppo-lpm':
                        error_pred_loss = model.error_predictor_loss(batch_size=min(batch_size, 128))
                        if error_pred_loss is not None:
                            error_optimizer.zero_grad()
                            error_pred_loss.backward()
                            error_optimizer.step()
                
            with torch.no_grad():
                # use the last batch
                if args.algo == 'ppo-rnd':
                    model_pred_error = model.last_pred_error
                elif args.algo == 'ppo-lpm':
                    lpm_rnd_error = model.last_rnd_error
                    lpm_predicted_error = model.last_predicted_error
                    lpm_error_pred_loss = model.last_error_pred_loss
                    lpm_curiosity = model.last_lpm_curiosity
                elif args.algo == 'ppo-ngu':
                    ngu_inv_loss = model.last_inv_loss
                    ngu_episodic_reward = model.last_episodic_reward
                    if model.ngu_use_rnd:
                        ngu_rnd_loss = model.last_rnd_loss
                        ngu_lifelong_reward = model.last_lifelong_reward

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            
            if obs_rms is not None:
                ob_rms = obs_rms
            else:
                ob_rms = getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                
            print('Saving policy...')
            torch.save([
                actor_critic,
                ob_rms
            ], os.path.join(save_path, args.env_name + ".pt"))
        
        if j % args.log_interval == 0 and len(episode_rewards) > 0 and len(update_episode_returns) > 0:
            print("")
            
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {:.2e}, FPS {} \n Last {} episodes: mean reward {:.1f}±{:.1f} (best: {:.1f}), mean length {:d}±{:d} (best: {:d}).\n Recent episodes ({:d}): reward {:.1f}±{:.1f}, length {:.1f}±{:.1f}.\n DistEntropy: {:.2e} CriticLoss: {:.2e} ActorLoss: {:.2e}"
                .format(j, float(total_num_steps), int( args.num_steps * args.num_processes / (end - start)), len(episode_rewards), 
                        np.mean(episode_rewards), np.std(episode_rewards), best_return, int(np.mean(episode_lengths)), int(np.std(episode_lengths)), int(best_length),
                        len(update_episode_returns), np.mean(update_episode_returns), np.std(update_episode_returns), np.mean(update_episode_lengths), np.std(update_episode_lengths),
                        dist_entropy, value_loss, action_loss))
            if len(update_episode_returns) > 0:
                wandb.log({
                    'rewards/environment_returns': np.mean(update_episode_returns),
                    'rewards/episode_lengths': np.mean(update_episode_lengths),
                    'rewards/best_return': best_return,
                    'rewards/best_length': best_length
                }, step=total_num_steps)
            if train_model:
                print(" IntRew: {:.3f}±{:.3f} ExtRew: {:.3f}±{:.3f} RollRew: {:.3f}±{:.3f}; RollRet: {:.3f}±{:.3f}"
                    .format(curiosity.mean(), curiosity.std(), ext_reward.mean(), ext_reward.std(), rollouts.rewards.mean(), rollouts.rewards.std(), rollouts.returns.mean(), rollouts.returns.std() ))
                wandb.log({
                    'rewards/int_rewards': curiosity.mean().item(),
                    'rewards/ext_rewards': ext_reward.mean().item(),
                    'rewards/rollouts_rewards': rollouts.rewards.mean().item(),
                    'rewards/rollouts_returns': rollouts.returns.mean().item()
                }, step=total_num_steps)
                if args.algo == 'ppo-rnd':
                    print(" PredError: {:.3f}".format(model_pred_error))
                    wandb.log({
                        'model/prediction_error': model_pred_error
                    }, step=total_num_steps)
                elif args.algo == 'ppo-lpm':
                    print(" RNDError: {:.3f} PredError: {:.3f} ErrorLoss: {:.3f} LPMCuriosity: {:.3f}".format(
                        lpm_rnd_error, lpm_predicted_error, lpm_error_pred_loss, lpm_curiosity))
                    wandb.log({
                        'model/rnd_error': lpm_rnd_error,
                        'model/predicted_error': lpm_predicted_error,
                        'model/error_pred_loss': lpm_error_pred_loss,
                        'model/lpm_curiosity': lpm_curiosity,
                        'model/buffer_size': len(model.error_buffer)
                    }, step=total_num_steps)
                elif args.algo == 'ppo-ngu':
                    if model.ngu_use_rnd:
                        print(" InvLoss: {:.3f} EpisodicRew: {:.3f} RNDLoss: {:.3f} LifelongRew: {:.3f}".format(
                            ngu_inv_loss, ngu_episodic_reward, ngu_rnd_loss, ngu_lifelong_reward))
                        wandb.log({
                            'model/inv_loss': ngu_inv_loss,
                            'model/episodic_reward': ngu_episodic_reward,
                            'model/rnd_loss': ngu_rnd_loss,
                            'model/lifelong_reward': ngu_lifelong_reward
                        }, step=total_num_steps)
                    else:
                        print(" InvLoss: {:.3f} EpisodicRew: {:.3f}".format(
                            ngu_inv_loss, ngu_episodic_reward))
                        wandb.log({
                            'model/inv_loss': ngu_inv_loss,
                            'model/episodic_reward': ngu_episodic_reward
                        }, step=total_num_steps)
                else:
                    print(" ModelMSE: {:.3f}".format(model_mse))
            if args.env_name in ['MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:

                if args.env_name in ['MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving']:
                    print(" " + args.env_name)
                    pos_range = infos[-1]['pos_range']
                    print("  PosRange: ({:.3f}, {:.3f}) PosInterval: {:.3f} ({:.3f})"
                        .format(pos_range[0], pos_range[1], pos_range[1] - pos_range[0], (pos_range[1] - pos_range[0]) / 1.8))
                    vel_range = infos[-1]['vel_range']
                    print("  VelRange: ({:.3f}, {:.3f}) VelInterval: {:.3f} ({:.3f})"
                        .format(vel_range[0], vel_range[1], vel_range[1] - vel_range[0], (vel_range[1] - vel_range[0]) / 0.14))

                    wandb.log({
                        'mountain_car/pos_interval': (pos_range[1] - pos_range[0]) / 1.8,
                        'mountain_car/vel_interval': (vel_range[1] - vel_range[0]) / 0.14
                    }, step=total_num_steps)

                    from exploration.environments.mountain_car_sparse import rate_buffer_with_blocks
                    n_blocks = 10
                    coverage, blocks = rate_buffer_with_blocks(update_state_buffer, n_blocks=n_blocks)
                    print("  UpdateStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
                    wandb.log({
                        'mountain_car/update_coverage': (coverage / n_blocks**2 * 100)
                    }, step=total_num_steps)
                    update_state_buffer = []
                    if len(overall_blocks) == 0:
                        overall_blocks = blocks
                    else:
                        overall_blocks = np.unique(np.concatenate([blocks, overall_blocks], axis=0), axis=0)
                    coverage = len(overall_blocks) #  = rate_buffer_with_blocks(states_buffer, n_blocks=n_blocksoverall_blocks)
                    print("  OverallStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
                    wandb.log({
                        'mountain_car/overall_coverage': (coverage / n_blocks**2 * 100)
                    }, step=total_num_steps)

                elif args.env_name == 'HalfCheetahSparse-v3':
                    print(" HalfCheetahSparse-v3:")
                    angle_range = infos[-1]['angle_range']
                    print("  AngleVelRange: ({:.3f}, {:.3f}) AngleVelRange: {:.3f}"
                        .format(angle_range[0], angle_range[1], angle_range[1] - angle_range[0]))
                    pos_range = infos[-1]['pos_range']
                    print("  PosRange: ({:.3f}, {:.3f}) PosInterval: {:.3f}"
                        .format(pos_range[0], pos_range[1], pos_range[1] - pos_range[0]))
                    vel_range = infos[-1]['vel_range']
                    print("  VelRange: ({:.3f}, {:.3f}) VelInterval: {:.3f}"
                        .format(vel_range[0], vel_range[1], vel_range[1] - vel_range[0]))
                    wandb.log({
                        'half_cheetah/angle_vel_interval': angle_range[1] - angle_range[0],
                        'half_cheetah/pos_interval': pos_range[1] - pos_range[0],
                        'half_cheetah/vel_interval': vel_range[1] - vel_range[0]
                    }, step=total_num_steps)

                    from exploration.environments.half_cheetah_sparse import rate_buffer_with_blocks
                    n_blocks = 10
                    coverage, blocks = rate_buffer_with_blocks(update_state_buffer, n_blocks=n_blocks)
                    print("  UpdateStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
                    wandb.log({
                        'half_cheetah/update_coverage': (coverage / n_blocks**2 * 100)
                    }, step=total_num_steps)
                    update_state_buffer = []
                    if len(overall_blocks) == 0:
                        overall_blocks = blocks
                    else:
                        overall_blocks = np.unique(np.concatenate([blocks, overall_blocks], axis=0), axis=0)
                    coverage = len(overall_blocks)
                    print("  OverallStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
                    wandb.log({
                        'half_cheetah/overall_coverage': (coverage / n_blocks**2 * 100)
                    }, step=total_num_steps)
            if args.env_name == 'MagellanAnt-v2':
                print(" MagellanAnt-v2")
                from exploration.environments.magellan_ant import rate_buffer
                coverage, blocks = rate_buffer(update_state_buffer)
                print("  UpdateMazeCoverage: {} ({:.3f})"
                    .format(coverage, (coverage / 7 * 100)))
                wandb.log({
                    'ant_maze/update_coverage': (coverage / 7 * 100)
                }, step=total_num_steps)
                update_state_buffer = []
                if len(overall_blocks) == 0:
                    overall_blocks = blocks
                else:
                    overall_blocks = set([*list(blocks), *list(overall_blocks)]) 
                coverage = len(overall_blocks)
                print("  OverallMazeCoverage: {} ({:.3f})"
                    .format(coverage, (coverage / 7 * 100)))
                wandb.log({
                    'ant_maze/overall_coverage': (coverage / 7 * 100)
                }, step=total_num_steps)
            
            # Montezuma's Revenge coverage logging
            if 'Montezuma' in args.env_name:
                print(" MontezumaRevenge:")
                
                # Update interval statistics
                update_pos_count = len(update_positions)
                update_room_count = len(update_rooms)
                print("  Update: UniquePositions: {}, UniqueRooms: {}".format(update_pos_count, update_room_count))
                wandb.log({
                    'montezuma/update_positions': update_pos_count,
                    'montezuma/update_rooms': update_room_count
                }, step=total_num_steps)
                
                # Overall statistics
                overall_pos_count = len(all_positions)
                overall_room_count = len(all_rooms)
                print("  Overall: UniquePositions: {}, UniqueRooms: {}".format(overall_pos_count, overall_room_count))
                wandb.log({
                    'montezuma/overall_positions': overall_pos_count,
                    'montezuma/overall_rooms': overall_room_count
                }, step=total_num_steps)
                
                # Per-episode statistics (if any episodes completed in this update)
                if len(episode_pos_counts) > 0:
                    avg_ep_positions = np.mean(episode_pos_counts[-10:])  # Last 10 episodes
                    avg_ep_rooms = np.mean(episode_room_counts[-10:])
                    print("  PerEpisode (last 10): AvgPositions: {:.1f}, AvgRooms: {:.1f}".format(avg_ep_positions, avg_ep_rooms))
                    wandb.log({
                        'montezuma/avg_episode_positions': avg_ep_positions,
                        'montezuma/avg_episode_rooms': avg_ep_rooms
                    }, step=total_num_steps)
                
                # Reset update counters
                update_positions = set()
                update_rooms = set()
                
    if args.env_name in ['MagellanAnt-v2', 'MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
        np.save(log_dir + '/overall_buffer.npy', states_buffer)
        if args.env_name in ['MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
            np.save(log_dir + '/overall_blocks.npy', blocks)
    
    # Save Montezuma's Revenge coverage data
    if 'Montezuma' in args.env_name:
        import json
        # Convert numpy types to Python types for JSON serialization
        all_positions_list = [(int(x), int(y), int(room)) for x, y, room in all_positions]
        all_rooms_list = [int(room) for room in all_rooms]
        
        coverage_data = {
            'all_positions': all_positions_list,
            'all_rooms': sorted(all_rooms_list),
            'total_positions': len(all_positions),
            'total_rooms': len(all_rooms),
            'episode_pos_counts': episode_pos_counts,
            'episode_room_counts': episode_room_counts
        }
        with open(log_dir + '/montezuma_coverage.json', 'w') as f:
            json.dump(coverage_data, f, indent=2)
        print(f"\nMontezuma coverage data saved to {log_dir}/montezuma_coverage.json")
        print(f"Final statistics: {len(all_positions)} unique positions, {len(all_rooms)} unique rooms visited")

if __name__ == "__main__":
    main()
