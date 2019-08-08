from utils.config import cfg

import math
import numpy as np
import os


class TemporalSampler:
    def __init__(self, frame_sampling_method):
        self.frame_sampling_method = frame_sampling_method
        self.frame_offsets = np.arange(-math.floor(cfg.TEMPORAL_INPUT_SIZE / 4), math.floor(cfg.TEMPORAL_INPUT_SIZE / 4))[:, np.newaxis]

    def frame_sampler(self, in_nframes):
        if self.frame_sampling_method == 'uniform':
            num_frames = cfg.RST.NFRAMES_PER_VIDEO-1 if cfg.RST.NFRAMES_PER_VIDEO > 1 else cfg.RST.NFRAMES_PER_VIDEO
            sample_rate = max(math.floor((in_nframes - cfg.TEMPORAL_INPUT_SIZE / 2)/num_frames), 1)
            frame_samples = np.arange(int(cfg.TEMPORAL_INPUT_SIZE / 4), int(in_nframes-cfg.TEMPORAL_INPUT_SIZE / 4) + 1,
                                      sample_rate)

        elif self.frame_sampling_method == 'temporal_stride':
            frame_samples = np.arange(int(cfg.TEMPORAL_INPUT_SIZE / 4), int(in_nframes-cfg.TEMPORAL_INPUT_SIZE / 4) + 1,
                                      cfg.RST.TEMPORAL_STRIDE[0])
            if len(frame_samples) < cfg.RST.NFRAMES_PER_VIDEO:
                frame_samples = np.linspace(cfg.TEMPORAL_INPUT_SIZE / 4, in_nframes - cfg.TEMPORAL_INPUT_SIZE / 4,
                                            cfg.RST.NFRAMES_PER_VIDEO)
                frame_samples = frame_samples.round().tolist()

        elif self.frame_sampling_method == 'random':
            frame_samples = np.random.permutation(
                int(in_nframes - cfg.TEMPORAL_INPUT_SIZE / 2)
            ) + cfg.TEMPORAL_INPUT_SIZE / 4

        elif self.frame_sampling_method == 'temporal_stride_random':
            temporal_stride = np.random.randint(cfg.RST.TEMPORAL_STRIDE[0], cfg.RST.TEMPORAL_STRIDE[1])
            frame_samples = np.arange(int(cfg.TEMPORAL_INPUT_SIZE / 4), int(in_nframes-cfg.TEMPORAL_INPUT_SIZE / 4) + 1,
                                      temporal_stride)
            if frame_samples[-1] + cfg.TEMPORAL_INPUT_SIZE / 4 >= in_nframes:
                frame_samples[-1] -= 2  # to avoid bug: in optical flow directories, some have image - 1 frames.

        elif self.frame_sampling_method == 'f25':
            frame_samples = np.linspace(cfg.TEMPORAL_INPUT_SIZE / 4, in_nframes - cfg.TEMPORAL_INPUT_SIZE / 4, 25)

        else:
            raise NotImplementedError

        # check the under or over frame sample list length.
        if self.frame_sampling_method != 'f25' and len(frame_samples) < cfg.RST.NFRAMES_PER_VIDEO:
            if len(frame_samples) > cfg.RST.NFRAMES_PER_VIDEO:
                frame_samples = frame_samples[:len(frame_samples) - len(frame_samples) % cfg.RST.NFRAMES_PER_VIDEO]
            add_frames, difference = 0, cfg.RST.NFRAMES_PER_VIDEO - len(frame_samples)
            while difference > 0:
                next_len = len(frame_samples) + 1
                add_samples = np.linspace(cfg.TEMPORAL_INPUT_SIZE / 4, in_nframes - cfg.TEMPORAL_INPUT_SIZE / 4,
                                          next_len, endpoint=False)
                add_samples = add_samples.round().tolist()
                add_samples = add_samples[np.random.randint(1, len(add_samples), 1)[0]]
                if add_frames > 20:
                    frame_samples = np.linspace(cfg.TEMPORAL_INPUT_SIZE / 4, in_nframes - cfg.TEMPORAL_INPUT_SIZE / 4,
                                                cfg.RST.NFRAMES_PER_VIDEO)
                    frame_samples = frame_samples.round().tolist()
                else:
                    frame_samples = np.append(frame_samples, add_samples)
                difference = cfg.RST.NFRAMES_PER_VIDEO - len(frame_samples)
                add_frames += 1
            frame_samples = np.sort(frame_samples)
        if self.frame_sampling_method != 'f25' and len(frame_samples) > cfg.RST.NFRAMES_PER_VIDEO:
            start = np.random.randint(0, len(frame_samples)-cfg.RST.NFRAMES_PER_VIDEO)
            frame_samples = frame_samples[start:start+cfg.RST.NFRAMES_PER_VIDEO]

        if self.frame_sampling_method == 'random':
            frame_samples = np.sort(frame_samples)

        assert (frame_samples == np.sort(frame_samples)).all()
        if frame_samples[-1] + cfg.TEMPORAL_INPUT_SIZE / 4 >= in_nframes:
            frame_samples[-1] -= 2  # to avoid bug: in optical flow directories, some have image - 1 frames.

        return frame_samples

    @staticmethod
    def generate_frame_list(frame_samples):
        images_list = ['frame{:06d}.jpg'.format(int(i)+1) for i in frame_samples]   # to adapt to MATLAB indexing

        return images_list

    def generate_flow_list(self, name, frame_samples):
        frame_offsets_tile = np.tile(self.frame_offsets, len(frame_samples))
        frame_samples = np.repeat(
            np.asarray(frame_samples)[np.newaxis, :],
            int(cfg.TEMPORAL_INPUT_SIZE / 2),
            axis=0
        ) + frame_offsets_tile
        flows_list = []
        for i in range(frame_samples.shape[1]):
            flows_chunk = []
            for j in range(int(cfg.TEMPORAL_INPUT_SIZE / 2)):
                frame_name = 'frame{:06d}.jpg'.format(int(frame_samples[j, i]) + 1)  # to adapt to MATLAB indexing
                flows_chunk.append(os.path.join('u', name, frame_name))
                flows_chunk.append(os.path.join('v', name, frame_name))
            flows_list.append(flows_chunk)

        return flows_list

    def generate(self, name, in_nframes):
        frame_samples = self.frame_sampler(in_nframes)
        return self.generate_frame_list(frame_samples), self.generate_flow_list(name, frame_samples)

