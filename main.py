import tensorflow as tf
import jax
import flax
from flax import linen as nn
from jax import numpy as jnp
from jax import random
from einops import einsum, rearrange
import optax
import matplotlib.pyplot as plt
import pickle

import numpy as np
from flax.training import orbax_utils

import orbax.checkpoint


import os
import uuid
import datetime

from maze_dataset.plotting import MazePlot
from maze_dataset.tokenization.token_utils import strings_to_coords

from dataset import CustomMazeDataset
from dataset import NumpyLoader

from model import TransformerLM, TransformerConfig


def main():
    # config details
    checkpoint_path = "data/2023-10-31_16-24-46"
    base_path = "data"
    save = True

    np_seed = 0
    jnp_seed = 0

    batch_size = 128
    lr = 1e-4
    n_train_steps = 10000000

    save_every_n_steps = 1000
    keep_n_checkpoints = 100

    n_worker = 4

    # n_eval = 1024
    emb_dim: int = 256
    num_heads: int = 16
    num_layers: int = 12
    qkv_dim: int = 256  # 512
    mlp_dim: int = 1024  # 2048
    max_len = 256

    grid_n = 5


    @jax.jit
    def train_step(state, batch):
        params = state['params']
        opt_state = state['opt_state']
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        step = state['step'] + 1

        return {'params': params, 'opt_state': opt_state, 'loss': loss, 'step': step}

    @jax.jit
    def eval_step(state, batch):
        params = state['params']
        loss = loss_fn(params, batch)
        return loss

    dataset = CustomMazeDataset(include_maze=False)
    train_loader = NumpyLoader(dataset, batch_size=batch_size, num_workers=n_worker)

    losses = []
    eval_losses = []

    key = random.PRNGKey(jnp_seed)
    rng, key = random.split(key)

    config = TransformerConfig(
        vocab_size=dataset.vocab_size,
        output_vocab_size=dataset.vocab_size,
        max_len=max_len,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim
    )

    model = TransformerLM(config=config)

    def loss_fn(params, batch):
        preds, act = model.apply(params, batch['data'])
        preds = preds[:, 0:-1]
        targets = batch['data'][:, 1:]
        idx = jnp.arange(targets.shape[1])[None, :]
        mask = jnp.where((idx <= batch['end_index'][:, None]) & (idx >= batch['start_index'][:, None]), 1., 0.)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=preds,
            labels=targets
        ) * mask

        loss = loss.sum() / mask.sum()

        return loss

    tx = optax.adamw(lr)

    x = next(iter(train_loader))
    params = model.init(rng, x['data'])

    apply_fn = jax.jit(model.apply)

    opt_state = tx.init(params)

    state = {'params': params, 'opt_state': opt_state, 'loss': 0., 'step': 0}

    # checkpoint management / loading model

    if save and not checkpoint_path:
        # make new run dir ect

        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Create a directory name with the date and unique ID
        checkpoint_dir_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Create the full path for the checkpoint directory
        checkpoint_path = os.path.join(base_path, checkpoint_dir_name)

        # Check if the directory already exists
        if not os.path.exists(checkpoint_path):
            # Create the directory
            os.makedirs(checkpoint_path)
            print(f"Checkpoint directory created: {checkpoint_path}")
        else:
            print(f"Checkpoint directory already exists: {checkpoint_path}")

    if checkpoint_path:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=keep_n_checkpoints)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_path, orbax_checkpointer, options)

        dummy_dict = {
            'state': state}


        step = checkpoint_manager.latest_step()

        if step:
            print(f'loading step {step}')
            load_dict = checkpoint_manager.restore(step, items=dummy_dict)
            state = load_dict['state']
            loss = load_dict['loss']
            loss = list(loss)


    # training loop

    import time

    # do the actual training
    loop_time = time.time()
    # start_time = time.time()
    # end_time = time.time()


    for n, batch in enumerate(train_loader):

        if save and n % save_every_n_steps == 0:
            save_step = state['step']
            print(f'saving at step {save_step}')
            save_dict = {'state': state,
                         'loss': jnp.array(losses)
                         }
            save_args = orbax_utils.save_args_from_target(save_dict)
            checkpoint_manager.save(save_step, save_dict, save_kwargs={'save_args': save_args})


        if n >= n_train_steps:
            break
        # print('non train step stuff: {:.5f}'.format(loop_time-old_time-end_time+start_time))
        # del batch['maze']

        # with open(path.join(PATH,'train_state_step_{}.p'.format(state['opt_state'][-1].count)), 'wb') as fp:
        #  pickle.dump(state, fp)

        # start_time = time.time()
        state = train_step(state, batch)
        # [insert code for Part 1 here]
        # end_time = time.time()
        # print("Time for Train step: {:.5f} seconds".format(end_time - start_time))

        losses.append(state['loss'])

        old_time = loop_time
        loop_time = time.time()
        print('steps per second: {:.5f}'.format(1/(loop_time - old_time)))
        print(f'step: {state["step"]}')
        print('loss: {}'.format(state['loss']))

        # eval_loss = eval_step(state, eval_batch)
        # print(f'eval_loss: {eval_loss}')
        # eval_losses.append(eval_loss)


if __name__ == '__main__':
    main()