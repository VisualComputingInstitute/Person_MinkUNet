import datetime
import signal
import tqdm

import torch
from torch.nn.utils import clip_grad_norm_


def _tqdm_str(t_dict):
    n = t_dict["total"]
    t = t_dict["elapsed"]
    r = t / n
    return "{}, elapsed={}, rate={:.2f}s/it, total_it={}".format(
        t_dict["prefix"], str(datetime.timedelta(seconds=int(t))), r, n,
    )


class Trainer(object):
    def __init__(self, logger, optimizer, cfg):
        self._logger = logger
        self._optim = optimizer
        self._epoch, self._step = 0, 0

        self._grad_norm_clip = cfg["grad_norm_clip"]
        self._ckpt_interval = cfg["ckpt_interval"]
        self._eval_interval = cfg["eval_interval"]
        self._max_epoch = cfg["epoch"]

        self.__sigterm = False
        signal.signal(signal.SIGINT, self._sigterm_cb)
        signal.signal(signal.SIGTERM, self._sigterm_cb)

    def set_epoch_step(self, epoch=None, step=None):
        if epoch is not None:
            self._epoch = epoch
        if step is not None:
            self._step = step

    def evaluate(
        self,
        model,
        eval_loader,
        tb_prefix,
        full_eval=False,
        plotting=False,
        rm_files=False,
    ):
        model.eval()
        tb_dict_list = []
        eval_dict_list = []
        split = eval_loader.dataset.split
        pbar = tqdm.tqdm(
            total=len(eval_loader),
            leave=False,
            desc=f"eval ({split}) ({self._epoch}/{self._max_epoch})",
        )

        # prevent out-of-memory during evaluation
        torch.cuda.empty_cache()

        for b_idx, batch in enumerate(eval_loader):
            if self.__sigterm:
                pbar.close()
                self._logger.log_info(_tqdm_str(pbar.format_dict))
                return 1

            with torch.no_grad():
                tb_dict, eval_dict = self._evaluate_batch(
                    model,
                    batch,
                    full_eval,
                    plotting,
                    self._logger.get_save_dir(self._epoch, split),
                    raise_oom=False,
                )

                # collate tb_dict for epoch evaluation
                tb_dict_list.append(tb_dict)
                eval_dict_list.append(eval_dict)

                # # save file
                # for k, v in file_dict.items():
                #     self._logger.save_file(v, k, self._epoch, split)

                # # save figure
                # for k, (fig, ax) in fig_dict.items():
                #     # self._logger.add_fig(k, fig, self._step)
                #     self._logger.save_fig(fig, k, self._epoch, split, close_fig=True)

                pbar.update()

        pbar.close()
        self._logger.log_info(_tqdm_str(pbar.format_dict))

        # prevent out-of-memory during evaluation
        torch.cuda.empty_cache()

        tb_dict = model.model_eval_collate_fn(
            tb_dict_list,
            eval_dict_list,
            self._logger.get_save_dir(self._epoch, split),
            full_eval=full_eval,
            rm_files=rm_files,
        )

        for k, v in tb_dict.items():
            self._logger.add_scalar(f"{tb_prefix}_{k}", v, self._step)

        # for k, v in epoch_dict.items():
        #     self._logger.save_dict(v, k, self._epoch, split)

        return 0

    def train(self, model, train_loader, eval_loader=None):
        # for self._epoch in tqdm.trange(0, self._max_epoch, desc="epochs"):
        for self._epoch in range(self._max_epoch):
            if self.__sigterm:
                self._logger.save_sigterm_ckpt(
                    model, self._optim, self._epoch, self._step,
                )
                return 1

            self._train_epoch(model, train_loader)

            if not self.__sigterm:
                do_ckpt = self._is_ckpt_epoch()
                do_eval = eval_loader is not None and (
                    self._is_evaluation_epoch() or do_ckpt
                )

                if do_ckpt:
                    self._logger.save_ckpt(
                        f"ckpt_e{self._epoch}.pth",
                        model,
                        self._optim,
                        self._epoch,
                        self._step,
                    )

                if do_eval:
                    self.evaluate(
                        model,
                        eval_loader,
                        tb_prefix="VAL",
                        full_eval=do_ckpt,
                        plotting=False,
                        rm_files=True,
                    )

            self._logger.flush()

        return 0

    def _is_ckpt_epoch(self):
        return self._epoch % self._ckpt_interval == 0 or self._epoch == self._max_epoch

    def _is_evaluation_epoch(self):
        return self._epoch % self._eval_interval == 0 or self._epoch == self._max_epoch

    def _sigterm_cb(self, signum, frame):
        self.__sigterm = True
        self._logger.log_info(f"Received signal {signum} at frame {frame}.")

    def _train_batch(self, model, batch, ratio, raise_oom, reduce_batch):
        """Train one batch. `ratio` in between [0, 1) is the progress of training
        current epoch. It is used by the scheduler to update learning rate.
        """
        model.train()
        self._optim.zero_grad()
        self._optim.set_lr(self._epoch + ratio)

        # https://github.com/pytorch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L283  # noqa
        try:
            loss, tb_dict, _ = model.model_fn(model, batch, reduce_batch=reduce_batch)
            loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e) and not raise_oom:
                reduce_batch = True
                self._logger.log_warning(
                    "Trainer: out of memory, retrying batch "
                    f"(reduce_batch: {reduce_batch})"
                )
                torch.cuda.empty_cache()
                return self._train_batch(
                    model, batch, ratio, raise_oom=True, reduce_batch=reduce_batch
                )
            else:
                self._logger.log_error(model.error_fn(model, batch))
                self._logger.save_ckpt(
                    f"ckpt_e{self._epoch}_error.pth",
                    model,
                    self._optim,
                    self._epoch,
                    self._step,
                )
                raise e

        if self._grad_norm_clip > 0:
            clip_grad_norm_(model.parameters(), self._grad_norm_clip)

        self._optim.step()

        self._logger.add_scalar("TRAIN_lr", self._optim.get_lr(), self._step)
        # self._logger.add_scalar("TRAIN_epoch", self._epoch + ratio, self._step)
        for key, val in tb_dict.items():
            self._logger.add_scalar(f"TRAIN_{key}", val, self._step)

        return loss.item()

        # # NOTE Dirty fix to use mixup regularization, to be removed
        # if model.mixup_alpha <= 0.0:
        #     return loss.item()

        # original_loss_value = loss.item()

        # self._optim.zero_grad()
        # loss, tb_dict, _ = model.model_fn_mixup(model, batch)
        # loss.backward()

        # if self._grad_norm_clip > 0:
        #     clip_grad_norm_(model.parameters(), self._grad_norm_clip)

        # self._optim.step()

        # for key, val in tb_dict.items():
        #     self._logger.add_scalar(f"TRAIN_{key}", val, self._step)

        # return original_loss_value

    def _train_epoch(self, model, train_loader):
        pbar = tqdm.tqdm(
            total=len(train_loader),
            leave=False,
            desc=f"train ({self._epoch}/{self._max_epoch})",
        )
        for ib, batch in enumerate(train_loader):
            if self.__sigterm:
                pbar.close()
                self._logger.log_info(_tqdm_str(pbar.format_dict))
                return

            loss = self._train_batch(
                model,
                batch,
                ratio=(ib / len(train_loader)),
                raise_oom=False,
                reduce_batch=False,
            )
            self._step += 1
            pbar.set_postfix({"total_it": self._step, "loss": loss})
            pbar.update()

        self._epoch = self._epoch + 1
        pbar.close()
        self._logger.log_info(_tqdm_str(pbar.format_dict))

    def _evaluate_batch(self, model, batch, full_eval, plotting, output_dir, raise_oom):
        try:
            return model.model_eval_fn(
                model,
                batch,
                full_eval=full_eval,
                plotting=plotting,
                output_dir=output_dir,
            )
        except RuntimeError as e:
            if "out of memory" in str(e) and not raise_oom:
                self._logger.log_warning("Trainer: out of memory, retrying batch")
                torch.cuda.empty_cache()
                return self._evaluate_batch(
                    model, batch, full_eval, plotting, output_dir, raise_oom=True
                )
            else:
                self._logger.log_error(model.error_fn(model, batch))
                self._logger.save_ckpt(
                    f"ckpt_e{self._epoch}_error.pth",
                    model,
                    self._optim,
                    self._epoch,
                    self._step,
                )
                raise e
