from torch.multiprocessing import Process, Pipe
import pickle
import cloudpickle
import numpy as np

def worker(remote, parent_remote, env_function_wrapper):
    parent_remote.close()
    env = env_function_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError



class CloudPickleWrapper(object):
    """
    Use CloudPickle to serialize contents otherwise multiprocessing uses pickle to serialize
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


class SubProcVecEnv(object):

    def __init__(self, envs):
        """

        :param envs: List of gym environments to run in subprocess
        """
        self.envs = envs
        self.closed = False

        # Get the number of envs to run
        num_envs = len(envs)
        # Create the remotes and work remotes accordingly
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        # Now create the different processes accordingly
        ps = [Process(target=worker, args=(worker_remote, remote, CloudPickleWrapper(env)))
              for (worker_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]
        # Iterate through the processes and start them
        for p in ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()

        # Close the work remotes
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()



    # Stepping into multiple environments and aggregating the results
    def step(self, actions):
        # Send the corresponding action step for the remote
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        # Get the results from the remotes
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    # Reset the multiple enviroments
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    # Reset the tasks in the multiple environments
    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    # Close the environment
    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)

