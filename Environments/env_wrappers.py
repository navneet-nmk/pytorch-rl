"""

Environment wrappers to help the training - This is specifically used for training on the Mario
environment used for the Empowerment driven models.

"""

import numpy as np
from collections import deque
from PIL import Image
from gym.spaces.box import Box
import gym
import time, sys
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping.
    n is the length of the buffer, and number of observations stacked.
    skip is the number of steps between buffered observations (min=1).
    n.b. first obs is the oldest, last obs is the newest.
         the buffer is zeroed out on reset.
         *must* call reset() for init!
    """
    def __init__(self, env=None, n=4, skip=4, shape=(84, 84),
                    channel_last=True, maxFrames=True):
        super(BufferedObsEnv, self).__init__(env)
        self.obs_shape = shape
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.maxFrames = maxFrames
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
        shape = shape + (n,) if channel_last else (n,) + shape
        self.observation_space = Box(0.0, 255.0, shape)
        self.ch_axis = -1 if channel_last else 0
        self.scale = 1.0 / 255
        self.observation_space.high[...] = 1.0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _observation(self, obs):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs = self._convert(self.env.reset())
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _convert(self, obs):
        self.obs_buffer.append(obs)
        if self.maxFrames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(Image.fromarray(intensity_frame).resize(
            self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame

    def _rgb2y(self, im):
        """Converts an RGB image to a Y image (as in YUV).
        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        if len(im.shape) < 3:
            return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)


class NoNegativeRewardEnv(gym.RewardWrapper):
    """Clip reward in negative direction."""
    def __init__(self, env=None, neg_clip=0.0):
        super(NoNegativeRewardEnv, self).__init__(env)
        self.neg_clip = neg_clip

    def _reward(self, reward):
        new_reward = self.neg_clip if reward < self.neg_clip else reward
        return new_reward


class SkipEnv(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def _step(self, action):
        total_reward = 0
        for i in range(0, self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info['steps'] = i + 1
            if done:
                break
        return obs, total_reward, done, info

class MarioEnv(gym.Wrapper):
    """
        Reset mario environment without actually restarting fceux everytime.
        This speeds up unrolling by approximately 10 times.
    """

    def __init__(self, env=None, tilesEnv=None):
        super(MarioEnv, self).__init__(env)
        self.resetCount = -1
        self.maxDistance = 3000
        self.tilesEnv = tilesEnv

    def _reset(self):
        if self.resetCount < 0:
            print('\nDoing hard mario fceux reset (40 seconds wait) !')
            sys.stdout.flush()
            self.env.reset()
            time.sleep(40)
        obs, _, _, info = self.env.step(7)  # take right once to start game
        if info.get('ignore', False):  # assuming this happens only in beginning
            self.resetCount = -1
            self.env.close()
            return self._reset()
        self.resetCount = info.get('iteration', -1)
        if self.tilesEnv:
            return obs
        return obs[24:-12, 8:-8, :]

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # print('info:', info)
        done = info['iteration'] > self.resetCount
        reward = float(reward) / self.maxDistance  # note: we do not use this rewards at all.
        if self.tilesEnv:
            return obs, reward, done, info
        return obs[24:-12, 8:-8, :], reward, done, info

    def _close(self):
        self.resetCount = -1
        return self.env.close()


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        """Warp frames to wxh as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

    
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_pytorch(env):
    return ImageToPyTorch(env)


