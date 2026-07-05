from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

import logging
import os


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len      在这里改了输入序列长
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in    在这里改了
        self.args.num_class = len(np.unique(test_data.y))
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        wd = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        return criterion

    def _save_debug_visualizations(self, loader, folder_path, split_name="test"):
        if not getattr(self.args, "viz_debug", False):
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[viz] matplotlib import failed, skip visualization: {e}")
            return

        max_samples = max(1, int(getattr(self.args, "viz_max_samples", 1)))
        fig_dir = os.path.join(folder_path, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        if self.swa:
            self.swa_model.eval()
            model_for_viz = self.swa_model
        else:
            self.model.eval()
            model_for_viz = self.model

        saved = 0
        with torch.no_grad():
            for batch_x, label, padding_mask in loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(device=self.device, dtype=torch.long)

                outputs = model_for_viz(batch_x, padding_mask, None, None)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                bs = batch_x.size(0)
                for b in range(bs):
                    if saved >= max_samples:
                        break

                    x_np = batch_x[b].detach().cpu().numpy()  # (T, C)
                    mask_np = padding_mask[b].detach().cpu().numpy()  # (T,)
                    prob_np = probs[b].detach().cpu().numpy()
                    pred_cls = int(preds[b].item())
                    true_cls = int(label[b].item())

                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    ax1, ax2, ax3, ax4 = axes.flatten()

                    im = ax1.imshow(x_np.T, aspect="auto", cmap="viridis", origin="lower")
                    ax1.set_title("x_enc heatmap (channel x time)")
                    ax1.set_xlabel("time")
                    ax1.set_ylabel("channel")
                    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

                    ax2.plot(mask_np, color="tab:orange", linewidth=1.5)
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.set_title("padding_mask (1=valid, 0=pad)")
                    ax2.set_xlabel("time")
                    ax2.set_ylabel("mask")

                    mean_signal = x_np.mean(axis=1)
                    ax3.plot(mean_signal, color="tab:blue", linewidth=1.2)
                    ax3.fill_between(
                        np.arange(len(mask_np)),
                        mean_signal.min(),
                        mean_signal.max(),
                        where=(mask_np < 0.5),
                        alpha=0.15,
                        color="gray",
                    )
                    ax3.set_title("mean signal over T")
                    ax3.set_xlabel("T", fontname="Arial", fontstyle="italic")
                    ax3.set_ylabel("C", fontname="Arial", fontstyle="italic")
                    ax3.set_xticks([])
                    ax3.set_yticks([])
                    ax3.spines["top"].set_visible(False)
                    ax3.spines["right"].set_visible(False)
                    ax3.spines["left"].set_visible(True)
                    ax3.spines["bottom"].set_visible(True)
                    # Add arrowheads to Cartesian axes.
                    ax3.annotate(
                        "",
                        xy=(1.02, 0.0),
                        xytext=(0.0, 0.0),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
                        clip_on=False,
                    )
                    ax3.annotate(
                        "",
                        xy=(0.0, 1.02),
                        xytext=(0.0, 0.0),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
                        clip_on=False,
                    )

                    cls_idx = np.arange(len(prob_np))
                    colors = ["tab:blue"] * len(prob_np)
                    colors[true_cls] = "tab:green"
                    colors[pred_cls] = "tab:red"
                    ax4.bar(cls_idx, prob_np, color=colors, alpha=0.85)
                    ax4.set_ylim(0.0, 1.0)
                    ax4.set_title(f"class probs (true={true_cls}, pred={pred_cls})")
                    ax4.set_xlabel("class index")
                    ax4.set_ylabel("probability")

                    fig.suptitle(
                        f"{split_name} sample#{saved} | model={self.args.model} | setting={self.args.model_id}",
                        fontsize=10,
                    )
                    fig.tight_layout()

                    save_path = os.path.join(
                        fig_dir,
                        f"{split_name}_sample_{saved}_true_{true_cls}_pred_{pred_cls}.png",
                    )
                    fig.savefig(save_path, dpi=150)
                    plt.close(fig)
                    saved += 1

                if saved >= max_samples:
                    break

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                # CrossEntropyLoss 需要 target 为 Long（类别索引），且必须在同一 device
                label = label.to(device=self.device, dtype=torch.long)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                # 兼容某些模型在训练/前向时返回 (logits, extra_loss) 的情况
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict



    # 在train方法外部设置logging


    # 在train方法中使用 logger
    def train(self, setting,logger):
        # 设置日志文件路径
        # log_file = './train_log.txt'
        # logger = setup_logger(log_file)
        # print('11111111111111111111111111111111111111111111')
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        # print('11111111111111111111111111111111111111111111')

    

        path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
        )

        # 构建日志文件路径
        # log_dir = './log/' + self.args.task_name + "/" + self.args.model_id + "/" + self.args.model+'/'+ setting + "/"
        #
        # # 创建日志目录（如果不存在）
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        #
        # # 最终的日志文件路径
        # log_file = os.path.join(log_dir, 'log.txt')
        #
        # # 现在你可以将日志文件传递给 logger 或其他操作
        #
        # logger = setup_logger(log_file)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        logger.info("train_steps: %d", train_steps)

        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5,logger=logger
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            logger.info('')  # 记录一个空行
            logger.info('')  # 记录一个空行
            epoch_time = time.time()
            

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                padding_mask = padding_mask.float().to(self.device)

                # CrossEntropyLoss 需要 target 为 Long（类别索引）
                label = label.to(self.device)


                # outputs = self.model(batch_x, padding_mask, None, None)
                # loss = criterion(outputs, label.long())
                output = self.model(batch_x, padding_mask, None, None)

                if isinstance(output, tuple):
                    # Mixup 被触发
                    logits, lam, idx = output
                    loss = lam * criterion(logits, label.long()) + (1 - lam) * criterion(logits, label[idx].long())
                else:
                    # Mixup 未触发 或 推理模式
                    logits = output
                    loss = criterion(logits, label.long())


# =================================
                _m = self.model.module if hasattr(self.model, "module") else self.model
                if getattr(_m, "auxiliary_loss", None) is not None:
                    loss = loss + getattr(
                        self.args, "diffusion_weight", 0.1
                    ) * _m.auxiliary_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info(
                        "\titers: %d, epoch: %d | loss: %.7f",
                        i + 1, epoch + 1, loss.item()
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info("\tspeed: %.4fs/iter; left time: %.4fs", speed, left_time)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            logger.info("Epoch: %d cost time: %.4f", epoch + 1, time.time() - epoch_time)
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            logger.info(
                "Epoch: %d, Steps: %d | Train Loss: %.5f\n"
                "Validation results --- Loss: %.5f, Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1: %.5f, AUROC: %.5f, AUPRC: %.5f\n"
                "Test results --- Loss: %.5f, Accuracy: %.5f, Precision: %.5f, Recall: %.5f, F1: %.5f, AUROC: %.5f, AUPRC: %.5f",
                epoch + 1, train_steps, train_loss, vali_loss,
                val_metrics_dict['Accuracy'], val_metrics_dict['Precision'], val_metrics_dict['Recall'],
                val_metrics_dict['F1'], val_metrics_dict['AUROC'], val_metrics_dict['AUPRC'],
                test_loss, test_metrics_dict['Accuracy'], test_metrics_dict['Precision'],
                test_metrics_dict['Recall'], test_metrics_dict['F1'], test_metrics_dict['AUROC'],
                test_metrics_dict['AUPRC']
            )

            es_m = getattr(self.args, "cls_early_stop_metric", "f1")
            if es_m == "f1":
                es_val = -val_metrics_dict["F1"]
            elif es_m == "auroc":
                es_val = -val_metrics_dict["AUROC"]
            elif es_m == "acc":
                es_val = -val_metrics_dict["Accuracy"]
            elif es_m == "precision":
                es_val = -val_metrics_dict["Precision"]
            elif es_m == "recall":
                es_val = -val_metrics_dict["Recall"]
            elif es_m == "auprc":
                es_val = -val_metrics_dict["AUPRC"]
            elif es_m == "mean6":
                mean6 = (
                    val_metrics_dict["Accuracy"]
                    + val_metrics_dict["Precision"]
                    + val_metrics_dict["Recall"]
                    + val_metrics_dict["F1"]
                    + val_metrics_dict["AUROC"]
                    + val_metrics_dict["AUPRC"]
                ) / 6.0
                es_val = -mean6
            elif es_m == "loss":
                # 验证 CE：EarlyStopping 内 score=-es_val，等价于最小化 loss
                es_val = vali_loss
            else:
                raise ValueError(
                    f"未知 cls_early_stop_metric={es_m!r}，可选: "
                    "f1, auroc, acc, precision, recall, auprc, mean6, loss"
                )
            early_stopping(
                es_val,
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, logger=None):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        if test:
            # 使用 logger 或 print
            if logger is not None:
                logger.info("loading model")
            else:
                print("loading model")

            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # 保存结果
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 使用 logger 或 print 输出验证和测试结果
        result_message = (
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )

        if logger is not None:
            logger.info(result_message)
        else:
            print(result_message)

        self._save_debug_visualizations(test_loader, folder_path, split_name="test")

        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        _m = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(_m, "export_tarcif_coeffs_text"):
            coeff_block = _m.export_tarcif_coeffs_text()
            f.write(coeff_block)
            if logger is not None:
                logger.info(coeff_block.rstrip())
            else:
                print(coeff_block.rstrip())
        f.write("\n")
        f.write("\n")
        f.close()
        return
