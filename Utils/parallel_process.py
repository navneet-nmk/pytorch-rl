from torch.multiprocessing import Process, Pipe
import pickle
import cloudpickle

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
        remotes, work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        # Now create the different processes accordingly
        

