import argparse
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="TimesNet")
    # parser.add_argument('--label_smoothing', type=float, default=0,
    #                     help='label smoothing for CrossEntropyLoss (0~0.2)')
    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)"
    )

    # model define for baselines
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16, help="chunk_size used in LightTS"
    )
    parser.add_argument(
        "--patch_len", type=int, default=16, help="patch_len used in PatchTST"
    )
    parser.add_argument("--stride", type=int, default=8, help="stride used in PatchTST")
    parser.add_argument(
        "--sampling_rate", type=int, default=256, help="frequency sampling rate"
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--resolution_list",
        type=str,
        default="2,4,8",
        help="MedGNN: comma-separated downsampling factors for multi-resolution branches",
    )
    parser.add_argument(
        "--nodedim",
        type=int,
        default=10,
        help="MedGNN MRGNN GraphLayer node embedding dimension",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip,shuffle,frequency,jitter,mask,drop",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )

    # 添加不同的卷积核心
    parser.add_argument(
        "--patch_H",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )

    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="classification: ratio of TRAIN subjects/samples to use (0,1], while keeping VAL/TEST fixed",
    )
    parser.add_argument(
        "--train_subset_seed",
        type=int,
        default=42,
        help="classification: random seed for train subset selection when train_ratio < 1.0",
    )
    parser.add_argument("--train_epochs", type=int, default=20, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="early stopping patience"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Adam 权重衰减（L2）；分类任务可试 1e-4~1e-2 缓解过拟合到验证被试",
    )
    parser.add_argument(
        "--cls_early_stop_metric",
        type=str,
        default="f1",
        choices=[
            "f1",
            "auroc",
            "acc",
            "precision",
            "recall",
            "auprc",
            "mean6",
            "loss",
        ],
        help="分类任务早停依据的验证指标：除 loss 外均为越大越好；loss 为验证 CE 越小越好；mean6 为 Accuracy/Precision/Recall/F1/AUROC/AUPRC 六项算术平均",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging",
        default=False,
    )

    parser.add_argument("--t", type=int, default=1, help="")
    parser.add_argument("--n", type=int, default=8, help="")
    parser.add_argument("--a", type=float, default=1, help="")
    parser.add_argument("--b", type=float, default=1, help="")
    parser.add_argument('--learnab', type=str2bool, default=True, help='Enable learnab')
    parser.add_argument(
        "--use_cif",
        type=str2bool,
        default=False,
        help="our 模型：是否启用首尾部 CIF 通道混合（默认 False，双流从原信号起步）",
    )
    parser.add_argument(
        "--cif_bidirectional",
        type=str2bool,
        default=True,
        help="保留参数（当前 TCN 未使用；our 模型已不使用 CIF 前后段混合）",
    )
    parser.add_argument(
        "--sn_sep",
        type=str2bool,
        default=False,
        help="若 True：CIF 后对输入做滑动平均信号/噪声分解，仅用平滑信号训练（见 utils/signal_noise.py）",
    )
    parser.add_argument(
        "--sn_kernel",
        type=int,
        default=5,
        help="sn_sep 时滑动平均窗口（奇数，内部可修正）",
    )
    parser.add_argument(
        "--msg_enhance",
        type=str2bool,
        default=False,
        help="保留参数（当前 MSConvFormer 使用 MA 可解释前处理 + DDPM 潜分支 + TCN 分类，不再使用 MSG）",
    )
    parser.add_argument(
        "--msg_per_channel",
        type=str2bool,
        default=False,
        help="MSG 的频域掩模是否按通道独立（更强表达、参数更多）",
    )
    parser.add_argument(
        "--diffusion_weight",
        type=float,
        default=0.1,
        help="our 模型：潜空间 DDPM ε-预测辅助损失权重（L = L_CE + λ L_diff）",
    )
    parser.add_argument(
        "--use_subject_norm",
        type=str2bool,
        default=True,
        help="MSConvFormer 等模型：是否启用每样本通道归一化（Subject InstanceNorm）",
    )
    parser.add_argument(
        "--use_mixup",
        type=str2bool,
        default=True,
        help="MSConvFormer 等模型：训练时是否启用 Manifold Mixup",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="MSConvFormer 等模型：mixup 的 Beta(alpha, alpha) 参数",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=0.5,
        help="MSConvFormer 等模型：每个 batch 触发 mixup 的概率",
    )

    # === MSConvFormer ablation switches ===
    parser.add_argument(
        "--no_ms_blocks",
        action="store_true",
        default=False,
        help="MSConvFormer 消融：禁用 MultiScaleTemporalBlock（仅保留 Transformer 全局建模）",
    )
    parser.add_argument(
        "--no_transformer",
        action="store_true",
        default=False,
        help="MSConvFormer 消融：禁用 TransformerEncoder（仅保留多尺度卷积局部建模）",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="attn",
        choices=["attn", "mean", "max"],
        help="MSConvFormer 消融：pooling 类型（attn=AttentionPool, mean=masked mean, max=masked max）",
    )
    parser.add_argument(
        "--viz_debug",
        type=str2bool,
        default=False,
        help="分类任务：是否导出 x_enc/padding_mask/预测概率/label 的可视化图",
    )
    parser.add_argument(
        "--viz_max_samples",
        type=int,
        default=1,
        help="分类任务：每次测试最多可视化多少个样本（默认 1）",
    )
    parser.add_argument(
        "--visualize_input",
        type=str2bool,
        default=False,
        help="MSConvFormer: 是否在 classification 前向中可视化输入 x_enc（B,T,C）",
    )
    parser.add_argument(
        "--visualize_sample_idx",
        type=int,
        default=0,
        help="MSConvFormer: 可视化 batch 中第几个样本",
    )
    parser.add_argument(
        "--visualize_max_channels",
        type=int,
        default=None,
        help="MSConvFormer: 最多可视化多少个通道；None 表示全部通道",
    )
    parser.add_argument(
        "--visualize_save_path",
        type=str,
        default=None,
        help="MSConvFormer: 输入可视化图片保存路径；为 None 时尝试弹窗显示",
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=6, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )
    return parser


def finalize_args(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    return args
