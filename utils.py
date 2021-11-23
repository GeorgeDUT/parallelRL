"""
Functions that use multiple times
"""
import random

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    # print("update net")
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def push_and_pull_reverse_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    # print("update net")
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = -lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

def push_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    # print("update net")
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad*(random.random()+1)
    opt.step()

    # pull global parameters
    #lnet.load_state_dict(gnet.state_dict())

def push_rand_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    # print("update net")
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        # gp._grad = lp.grad
        # grad = torch.randn(tuple(lp.grad.shape), dtype=torch.float32)
        # grad = torch.randn(tuple(lp.grad.shape), dtype=torch.float32)
        gp._grad = 2*(torch.rand_like(lp.grad) * 2 - 1) #生成-1到1的均匀分布的随机梯度
    opt.step()

    # pull global parameters
    # lnet.load_state_dict(gnet.state_dict())


def push_constant3_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    # print("update net")
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        # gp._grad = lp.grad
        # grad = torch.randn(tuple(lp.grad.shape), dtype=torch.float32)
        # grad = torch.randn(tuple(lp.grad.shape), dtype=torch.float32)
        # gp._grad = torch.ones_like(lp.grad) #梯度
        gp._grad = 3*torch.ones_like(lp.grad) + torch.rand_like(lp.grad)#梯度
    opt.step()

def push_constant_N_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma, N):
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = torch.ones_like(lp.grad) * N
    opt.step()


def push_uniform_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma, min_grad, max_grad):
    assert max_grad > min_grad, 'max_ grad needs bigger than min_grad'
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = torch.rand_like(lp.grad) * (max_grad - min_grad) + torch.ones_like(lp.grad) * min_grad
    opt.step()


def push_gaussian_grad(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    lnet.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = torch.randn_like(lp.grad)
    opt.step()


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )