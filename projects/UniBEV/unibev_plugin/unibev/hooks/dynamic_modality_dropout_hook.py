import os.path as osp
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.fileio import FileClient
from mmcv.utils.path import check_file_exist




@HOOKS.register_module()
class DynamicModalityDropoutHook(CheckpointHook):

    def __init__(self,
                 interval = 1,
                 by_epoch = True,
                 start = 0.5,
                 save_optimizer = True,
                 out_dir = None,
                 file_client_args = None,
                 **kwargs):
        super().__init__()
        self.interval = interval
        self.by_epoch = by_epoch
        self.start = start
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs
        self.file_client_args = file_client_args


    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args, self.out_dir)

        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        if 1 < self.start and self.start < runner._max_epochs:
            self.start_epoch = self.start
        elif 0 < self.start and self.start <= 1:
            self.start_epoch = round(self.start * runner._max_epochs)
        else:
            raise ValueError(
                f'Start epoch must be between 0 and the number of max epochs')

        runner.logger.info(f'The Late Stage Checkpoints will be saved from the epoch {self.start_epoch} '
                           f'to {self.out_dir} by {self.file_client.name}.')

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if runner.epoch + 1 >= self.start_epoch and self.every_n_epochs(runner, self.interval):
            checkpoint_name = 'epoch_{}.path'.format(runner.epoch+1)
            file_path = osp.join(self.out_dir, checkpoint_name)
            if not osp.isfile(file_path):
                runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
                self._save_checkpoint(runner)
