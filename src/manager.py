import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from torch.optim import lr_scheduler

from components import models
from components import datasets
from components import methods

from utils.logger import ExpLogger, TimeCheck
from utils.metric import SummationMeter, Metric

class DLManager:
    def __init__(self, device, args, cfg=None):
        self.args = args
        self.cfg = cfg
        self.device = device
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None

        if self.cfg is not None:
            self._init_from_cfg(cfg, device)

        if args.checkpoint_path is not None:
            self.current_epoch = torch.load(args.checkpoint_path)['epoch']
        else:
            self.current_epoch = 0

    def _init_from_cfg(self, cfg, device):
        assert cfg is not None
        self.cfg = cfg

        self.model = _prepare_model(self.cfg.MODEL, self.args, device,
                                    is_distributed=self.args.is_distributed,
                                    local_rank=self.args.local_rank if self.args.is_distributed else None)
        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.args, self.model)
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.args, self.optimizer)

        self.get_train_loader = getattr(datasets, self.cfg.DATASET.TRAIN.NAME).get_dataloader
        self.get_test_loader = getattr(datasets, self.cfg.DATASET.TEST.NAME).get_dataloader
        # self.get_val_loader = getattr(datasets, self.cfg.DATASET.VAL.NAME).get_dataloader

        self.method = getattr(methods, self.cfg.METHOD)

    def train(self):
        if self.args.is_master:
            self._log_before_train()
        train_loader = self.get_train_loader(args=self.args,
                                             dataset_cfg=self.cfg.DATASET.TRAIN,
                                             dataloader_cfg=self.cfg.DATALOADER.TRAIN,
                                             is_distributed=self.args.is_distributed)

        test_loader = self.get_test_loader(args=self.args,
                                          dataset_cfg=self.cfg.DATASET.TEST,
                                          dataloader_cfg=self.cfg.DATALOADER.TEST,
                                          is_distributed=self.args.is_distributed)

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()
        print('Start Training {}'.format(self.current_epoch))
        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch)
            train_log_dict = self.method.train(model=self.model,
                                               data_loader=train_loader,
                                               optimizer=self.optimizer,
                                               device = self.device,
                                               is_distributed=self.args.is_distributed,
                                               world_size=self.args.world_size)

            self.scheduler.step()
            self.current_epoch += 1
            self._log_after_epoch(epoch, time_checker, train_log_dict, 'train')

            if epoch % self.args.save_term == 0 and epoch != 0:
                eva_log = self.method.evaluation(model=self.model, data_loader=test_loader, device=self.device,
                                                 logger=self.logger, sequence_name='epoch{}'.format(epoch))
                self._log_after_epoch(epoch , time_checker, eva_log, 'val')


    def eva(self):
        eva_loader = self.get_test_loader(args=self.args,
                                             dataset_cfg=self.cfg.DATASET.TEST,
                                             dataloader_cfg=self.cfg.DATALOADER.TEST,
                                             is_distributed=self.args.is_distributed)

        time_checker = TimeCheck(self.cfg.TOTAL_EPOCH)
        time_checker.start()
        print('Start evaluation {}'.format(self.current_epoch))

        eva_log = self.method.evaluation(model=self.model,
                         data_loader=eva_loader, logger=self.logger, device = self.device, sequence_name = 'evaluation_{}'.format(self.cfg['DATASET']['TEST']['FOLDER']))

        for key in eva_log.keys():
            # log += ' | %s: %s' % (key, str(eva_log[key]))
            print('{}: {}'.format(key, str(eva_log[key])))



    def save(self, name):
        checkpoint = self._make_checkpoint()
        self.logger.save_checkpoint(checkpoint, name)

    def load(self, name):
        checkpoint = self.logger.load_checkpoint(name)


        self.model.module.load_state_dict(checkpoint['model'])

    def _make_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'args': self.args,
            'cfg': self.cfg,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        return checkpoint

    def _gather_log(self, log_dict):
        if log_dict is None:
            return None

        for key in log_dict.keys():
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                log_dict[key].all_gather(self.args.world_size)

        return log_dict

    def _log_before_train(self):
        self.logger.train()
        self.logger.save_args(self.args)
        self.logger.save_cfg(self.cfg)
        self.logger.log_model(self.model)
        self.logger.log_optimizer(self.optimizer)
        # print(os.path.abspath(__file__))
        # print(os.path.dirname(os.path.abspath(__file__)))
        self.logger.save_src(os.path.dirname(os.path.abspath(__file__)))

    def _log_after_epoch(self, epoch, time_checker, log_dict, part):
        # Calculate Time
        time_checker.update(epoch)

        # Log Time
        self.logger.write('Epoch: %d | time per epoch: %s | eta: %s' %
                          (epoch, time_checker.time_per_epoch, time_checker.eta))

        # Log Learning Process
        log = '%5s' % part
        for key in log_dict.keys():
            log += ' | %s: %s' % (key, str(log_dict[key]))
            self.logger.add_scalar('%s/%s' % (part, key), log_dict[key].value, epoch)
        self.logger.write(log=log)

        # Make Checkpoint
        checkpoint = self._make_checkpoint()

        # Save Checkpoint
        self.logger.save_checkpoint(checkpoint, 'final.pth')
        if epoch % self.args.save_term == 0:
            self.logger.save_checkpoint(checkpoint, '%d.pth' % epoch)


def _prepare_model(model_cfg, args, device, is_distributed=False, local_rank=None):
    name = model_cfg.NAME
    parameters = model_cfg.PARAMS

    model = getattr(models, name)(**parameters)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        if args.checkpoint_path is not None:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            checkpoint = checkpoint['model']
            model.load_state_dict(checkpoint)
        model = nn.DataParallel(model).to(device)

    return model


def _prepare_optimizer(optimizer_cfg, args, model):
    name = optimizer_cfg.NAME
    parameters = optimizer_cfg.PARAMS
    learning_rate = parameters.lr

    # params_group = model.module.get_params_group(learning_rate)

    optimizer = getattr(optim, name)(model.parameters(), **parameters)

    if args.checkpoint_path is not None:
        optimizer.load_state_dict(torch.load(args.checkpoint_path)['optimizer'])

    return optimizer


def _prepare_scheduler(scheduler_cfg, args, optimizer):
    name = scheduler_cfg.NAME
    parameters = scheduler_cfg.PARAMS

    if name == 'CosineAnnealingWarmupRestarts':
        from utils.scheduler import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **parameters)
    else:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[100,200, 300, 400, 500], gamma=0.9)
        # scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

    if args.checkpoint_path is not None:
        scheduler.load_state_dict(torch.load(args.checkpoint_path)['scheduler'])

    return scheduler
