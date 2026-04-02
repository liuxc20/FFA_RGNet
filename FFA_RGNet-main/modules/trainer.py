import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer(object):
    def __init__(self, model,loss_fns, m_metric_ftns, ml_metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_fns = loss_fns
        self.m_metric_ftns = m_metric_ftns
        self.ml_metric_ftns = ml_metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'
        self.best_recorder['val']['model_name'] = self.args.model_name
        self.best_recorder['test']['model_name'] = self.args.model_name

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        #record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        #record_table = record_table.append(self.best_recorder['test'].cpu(), ignore_index=True)

        record_table = record_table.append({k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in self.best_recorder['val'].items()}, ignore_index=True)
        record_table = record_table.append({k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in self.best_recorder['test'].items()}, ignore_index=True)

        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        if epoch % 10 == 0: #每10个epoch，保存一个checkpoint，也能确保保存到了最后一个epoch
            filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            if epoch >4: #####从第8个opoch后才开始存储current best model
                best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                torch.save(state, best_path)
                print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))

    def _process_loss(self,phase_logits, location_logits, lesion_logits, diagnosis_logits,
                            phase_labels, location_labels, lesion_labels, diagnosis_labels,
                            mask,
                            loss_fns):
        """
        Args:
            phase_logits: (B, T, 3)
            location_logits: (B, T, 7)
            lesion_logits: (B, T, 8)
            diagnosis_logits: (B, 10)
            phase_labels: (B, T)
            location_labels: (B, T)
            lesion_labels: (B, T, 8)
            diagnosis_labels: (B,)
            mask: (B, T) bool tensor, True为有效帧
            loss_fns: dict with keys: 'phase', 'location', 'lesion', 'diagnosis'
            loss_weights: dict with keys: 'phase', 'location', 'lesion', 'diagnosis'

        Returns:
            total_loss:
            loss_dict: 
        """
        # if loss_weights is None:
        #     loss_weights = {'phase': 1.0, 'location': 1.0, 'lesion': 1.0, 'diagnosis': 1.0}

        B, T, _ = phase_logits.shape
        valid_mask = mask.view(B * T)

        phase_logits_flat = phase_logits.view(B * T, -1)
        location_logits_flat = location_logits.view(B * T, -1)
        lesion_logits_flat = lesion_logits.view(B * T, -1)

        phase_labels_flat = phase_labels.view(B * T)
        location_labels_flat = location_labels.view(B * T)
        lesion_labels_flat = lesion_labels.view(B * T, -1)

        phase_loss = loss_fns['phase'](phase_logits_flat[valid_mask], phase_labels_flat[valid_mask])
        location_loss = loss_fns['location'](location_logits_flat[valid_mask], location_labels_flat[valid_mask])
        lesion_loss = loss_fns['lesion'](lesion_logits_flat[valid_mask], lesion_labels_flat[valid_mask])
        diagnosis_loss = loss_fns['diagnosis'](diagnosis_logits, diagnosis_labels)

        total_loss = (
            phase_loss*1.0 +
            location_loss*1.0 +
            lesion_loss*1.5 +
            diagnosis_loss*2.0
        )

        loss_dict = {
            'phase': phase_loss.item(),
            'location': location_loss.item(),
            'lesion': lesion_loss.item(),
            'diagnosis': diagnosis_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict




class Trainer(BaseTrainer):
    def __init__(self, model,  loss_fns, m_metric_ftns, ml_metric_ftns,optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model,  loss_fns, m_metric_ftns, ml_metric_ftns,optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.writer = SummaryWriter(log_dir=args.save_dir)

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        epoch_loss_dict = defaultdict(list)

        for batch_idx, batch in enumerate(self.train_dataloader):
            # get data
            images = batch['images'].to(self.device)
            mask = batch['mask'].to(self.device)
            phases = batch['phases'].to(self.device)
            locations = batch['locations'].to(self.device)
            lesions = batch['lesions'].to(self.device)
            diagnosis = batch['diagnosis'].to(self.device)
            # forward
            phase_logits, location_logits, lesion_logits, diagnosis_logits, confidence, normalized_entropy = self.model(images, mask) 
            # compute loss
            loss, loss_dict = self._process_loss(
                phase_logits, location_logits, lesion_logits, diagnosis_logits,
                phases, locations, lesions, diagnosis,
                mask,
                self.loss_fns
                            )
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            for key in loss_dict:
                epoch_loss_dict[key].append(loss_dict[key])


        log = {'train_loss': train_loss / len(self.train_dataloader)}

        for key in epoch_loss_dict:
            log[f'train_{key}_loss'] = np.mean(epoch_loss_dict[key])
        self.writer.add_scalar('Loss/train_total', log['train_loss'], epoch)
        self.writer.add_scalar('Loss/train_phase', log['train_phase_loss'], epoch)
        self.writer.add_scalar('Loss/train_location', log['train_location_loss'], epoch)
        self.writer.add_scalar('Loss/train_lesion', log['train_lesion_loss'], epoch)
        self.writer.add_scalar('Loss/train_diagnosis', log['train_diagnosis_loss'], epoch)
  

        val_loss = 0
        self.model.eval()
        epoch_val_loss_dict = defaultdict(list)

        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, batch in enumerate(self.val_dataloader):
                images = batch['images'].to(self.device)
                mask = batch['mask'].to(self.device)
                phases = batch['phases'].to(self.device)
                locations = batch['locations'].to(self.device)
                lesions = batch['lesions'].to(self.device)
                diagnosis = batch['diagnosis'].to(self.device)
                phase_logits, location_logits, lesion_logits, diagnosis_logits, confidence, normalized_entropy= self.model(images, mask) 

                loss, loss_dict = self._process_loss(
                    phase_logits, location_logits, lesion_logits, diagnosis_logits,
                    phases, locations, lesions, diagnosis,
                    mask,
                    self.loss_fns
                )
                val_loss += loss.item()
                val_res.extend(diagnosis_logits)
                val_gts.extend(diagnosis)
                for key in loss_dict:
                    epoch_val_loss_dict[key].append(loss_dict[key])


            val_met = self.m_metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update({'val_loss': val_loss / len(self.val_dataloader)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

            for key in epoch_val_loss_dict:
                log[f'val_{key}_loss'] = np.mean(epoch_val_loss_dict[key])            

            self.writer.add_scalar('Loss/val_total', log['val_loss'], epoch)
            self.writer.add_scalar('Loss/val_phase', log['val_phase_loss'], epoch)
            self.writer.add_scalar('Loss/val_location', log['val_location_loss'], epoch)
            self.writer.add_scalar('Loss/val_lesion', log['val_lesion_loss'], epoch)
            self.writer.add_scalar('Loss/val_diagnosis', log['val_diagnosis_loss'], epoch)            

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res,lesion_res, lesion_gts = [], [],[], []
            lesion_res_list, lesion_gts_list = [], []
            for batch_idx, batch in enumerate(self.test_dataloader):
                images = batch['images'].to(self.device)
                mask = batch['mask'].to(self.device)
                phases = batch['phases'].to(self.device)
                locations = batch['locations'].to(self.device)
                lesions = batch['lesions'].to(self.device)
                diagnosis = batch['diagnosis'].to(self.device)
                phase_logits, location_logits, lesion_logits, diagnosis_logits, confidence, normalized_entropy= self.model(images, mask) 

                test_res.extend(diagnosis_logits)
                test_gts.extend(diagnosis)

                lesion_pred = torch.sigmoid(lesion_logits) > 0.5 # (B, T, L)
                B, T, L = lesion_pred.shape
                valid = mask.view(-1) == 1  # (B*T,)
                lesion_pred_flat = lesion_pred.view(B * T, L)[valid]  # (N_valid, L)
                lesion_gt_flat = lesions.view(B * T, L)[valid]        # (N_valid, L)
                lesion_res_list.append(lesion_pred_flat.cpu())
                lesion_gts_list.append(lesion_gt_flat.cpu())


            lesion_res = torch.cat(lesion_res_list, dim=0).cpu().numpy()
            lesion_gts = torch.cat(lesion_gts_list, dim=0).cpu().numpy()
            lesion_met = self.ml_metric_ftns(lesion_gts, lesion_res)
            log.update(**{'lesion_' + k: v for k, v in lesion_met.items()})

            
            test_met = self.m_metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                       {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        
        self.lr_scheduler.step()

        return log
