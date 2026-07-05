import os
import torch
from models import  MSConvFormer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "MSConvFormer": MSConvFormer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # 可见 GPU 在 PyTorch 里总是从 cuda:0 编号；物理卡号通过 CUDA_VISIBLE_DEVICES 指定。
            # 若用户已设置 CUDA_VISIBLE_DEVICES（如 CUDA_VISIBLE_DEVICES=0），则不再覆盖。
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                if self.args.use_multi_gpu:
                    devs = self.args.devices.replace(" ", "")
                    os.environ["CUDA_VISIBLE_DEVICES"] = devs
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device("cuda:0")
            print(
                "Use GPU: cuda:0 (CUDA_VISIBLE_DEVICES=%s)"
                % os.environ.get("CUDA_VISIBLE_DEVICES", "")
            )
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
