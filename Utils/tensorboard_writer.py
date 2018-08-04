"""

A Tensorboard Summary writer for showing the training curves
in a pytorch training loop.

"""


from tensorboardX import SummaryWriter


class TensorboardWriter(object):

    def __init__(self):
        self.writer = SummaryWriter()

    def write(self, scalar, scalar_name, epoch):
        self.writer.add_scalar(scalar_name, scalar, epoch)

    def write_scalars(self, scalars, scalar_grp_name, epoch):
        assert type(scalars) == 'dict'
        self.writer.add_scalars(scalar_grp_name, scalars, epoch)

    def export(self, path, close_writer=True):
        self.writer.export_scalars_to_json(path)
        if close_writer:
            self.writer.close()