import os.path as osp
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.fileio import FileClient


@HOOKS.register_module()
class CheckpointLateStageHook(CheckpointHook):
    """Save checkpoints more frequently at the late stage of training besides the checkpoint hook.
    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        start (int | None, optional): Late stage checkpoint saving starting epoch or iteration.
            Default: None.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``.
            `Changed in version 1.3.16.`

        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
        """
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
