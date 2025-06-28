import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, TensorDataset

import time

import numpy as np
import torch.nn as nn


def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get("_attacks").values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.
    """

    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model)
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
            print("Failed to set device automatically, please try set_device() manual.")

        # Controls attack mode.
        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None

        # Controls when normalization is used.
        self.normalization_used = None
        self._normalization_applied = None
        if self.model.__class__.__name__ == "RobModel":
            self._set_rmodel_normalization_used(model)

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def _set_rmodel_normalization_used(self, model):
        r"""
        Set attack normalization for MAIR [https://github.com/Harry24k/MAIR].

        """
        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
        if (mean is not None) and (std is not None):
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            if isinstance(std, torch.Tensor):
                std = std.cpu().numpy()
            if (mean != 0).all() or (std != 1).all():
                self.set_normalization_used(mean, std)

    @wrapper_method
    def set_normalization_used(self, mean, std):
        self.normalization_used = {}
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used["mean"] = mean
        self.normalization_used["std"] = std
        self._set_normalization_applied(True)

    def normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return inputs * std + mean

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self.attack_mode = "default"
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode, quiet):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print("Attack mode is changed to '%s'." % mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted("targeted(custom)", quiet)
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self, quiet=False):
        r"""
        Set attack mode as targeted with random labels.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted("targeted(random)", quiet)
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        r"""
        Set attack mode as targeted with least likely labels.

        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
            num_classses (str): number of classes. (Default: False)

        """
        self._set_mode_targeted("targeted(least-likely)", quiet)
        assert kth_min > 0
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_by_label(self, quiet=False):
        r"""
        Set attack mode as targeted.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        .. note::
            Use user-supplied labels as target labels.
        """
        self._set_mode_targeted("targeted(label)", quiet)
        self._target_map_function = "function is a string"

    @wrapper_method
    def set_model_training_mode(
        self, model_training=False, batchnorm_training=False, dropout_training=False
    ):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if "BatchNorm" in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if "Dropout" in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()

    def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def to_type(inputs, type):
        r"""
        Return inputs as int if float is given.
        """
        if type == "int":
            if isinstance(inputs, torch.FloatTensor) or isinstance(
                inputs, torch.cuda.FloatTensor
            ):
                return (inputs * 255).type(torch.uint8)
        elif type == "float":
            if isinstance(inputs, torch.ByteTensor) or isinstance(
                inputs, torch.cuda.ByteTensor
            ):
                return inputs.float() / 255
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")
        return inputs

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        print(
            "- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t"
            % (progress, rob_acc, l2, elapsed_time),
            end=end,
        )

    @staticmethod
    def load(
        load_path,
        batch_size=128,
        shuffle=False,
        normalize=None,
        load_predictions=False,
        load_clean_inputs=False,
    ):
        save_dict = torch.load(load_path)
        keys = ["adv_inputs", "labels"]

        if load_predictions:
            keys.append("preds")
        if load_clean_inputs:
            keys.append("clean_inputs")

        if save_dict["save_type"] == "int":
            save_dict["adv_inputs"] = save_dict["adv_inputs"].float() / 255
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"].float() / 255
                )  # nopep8

        if normalize is not None:
            n_channels = len(normalize["mean"])
            mean = torch.tensor(normalize["mean"]).reshape(1, n_channels, 1, 1)
            std = torch.tensor(normalize["std"]).reshape(1, n_channels, 1, 1)
            save_dict["adv_inputs"] = (save_dict["adv_inputs"] - mean) / std
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"] - mean
                ) / std  # nopep8

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=shuffle)
        print(
            "Data is loaded in the following order: [%s]" % (", ".join(keys))
        )  # nopep8
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(inputs)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError(
                "target_map_function is not initialized by set_mode_targeted."
            )
        if self.attack_mode == "targeted(label)":
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def __call__(self, inputs, labels=None, *args, **kwargs):
        given_training = self.model.training
        self._change_model_mode(given_training)

        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            # adv_inputs = self.to_type(adv_inputs, self.return_type)

            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            # adv_inputs = self.to_type(adv_inputs, self.return_type)

        self._recover_model_mode(given_training)

        return adv_inputs

    def __repr__(self):
        info = self.__dict__.copy()

        del_keys = ["model", "attack", "supported_mode"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self.attack_mode
        info["normalization_used"] = (
            True if self.normalization_used is not None else False
        )

        return (
            self.attack
            + "("
            + ", ".join("{}={}".format(key, val) for key, val in info.items())
            + ")"
        )

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        attacks = self.__dict__.get("_attacks")

        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if items not in stack:
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = list(items.keys()) + list(items.values())
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items

        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get("_attacks").items():
                attacks[name + "." + subname] = subvalue



class APGD_CLIP(Attack):
    def __init__(
        self,
        clip_model,
        norm="Linf",
        eps=8/255,
        steps=10,
        n_restarts=1,
        seed=0,
        eot_iter=1,
        rho=0.75,
        targeted=False,
        verbose=False,
    ):
        super().__init__("APGD_CLIP", clip_model)
        self.clip_model = clip_model
        self.targeted = targeted
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.supported_mode = ["default"]
        
        # CLIP图像预处理参数（根据实际预处理配置）
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(self.device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(self.device)
        
    def normalize_clip(self, x):
        return (x - self.clip_mean) / self.clip_std
    
    def inverse_normalize_clip(self, x):
        return x * self.clip_std + self.clip_mean

    def forward(self, images, text_features):
        # 输入处理：将图像归一化到CLIP预处理空间
        images = self.normalize_clip(images.clone().detach().to(self.device))
        text_features = text_features.clone().detach().to(self.device)
        
        # 生成对抗样本（在归一化空间）
        adv_images = self.perturb(images, text_features)
        
        # 将对抗样本转换回原空间
        adv_images = self.inverse_normalize_clip(adv_images).clamp(0, 1)
        return adv_images

    # def check_oscillation(self, x, j, k, y5, k3=0.75):
    #     t = np.zeros(x.shape[1])
    #     for counter5 in range(k):
    #         t += x[j - counter5] > x[j - counter5 - 1]
    #     return t <= k * k3 * np.ones(t.shape)

    def check_oscillation(self, loss_history, current_step, window_size=5, threshold_ratio=0.75):
        """
        检查损失值是否在最近窗口内出现振荡
        参数:
            loss_history (np.ndarray): 形状为(steps, batch_size)的损失历史数组
            current_step (int): 当前步骤索引
            window_size (int): 检查的窗口大小（默认5步）
            threshold_ratio (float): 判定为振荡的阈值比例（默认0.75）
        
        返回:
            np.ndarray: 布尔数组，标记每个样本是否需要调整步长
        """
        # 确保历史记录足够长
        if current_step < window_size:
            return np.zeros(loss_history.shape[1], dtype=bool)
        
        # 提取最近window_size+1步的损失值（需要比较window_size次变化）
        recent_losses = loss_history[current_step - window_size : current_step + 1]
        
        # 计算每一步是否比前一步大
        increases = (recent_losses[1:] > recent_losses[:-1])  # 形状(window_size, batch_size)
        
        # 统计窗口内损失增加的次数
        increase_counts = np.sum(increases, axis=0)
        
        # 判断是否超过阈值比例
        should_reduce = increase_counts > (window_size * threshold_ratio)
        
        return should_reduce

    def dlr_loss(self, x, y):
        pass  # 不再使用DLR损失

    def attack_single_run(self, x_in, text_features):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        
        # 初始化参数
        steps_2, steps_min, size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        
        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device) - 1
            x_adv = x + self.eps * t / t.reshape(x.shape[0], -1).abs().max(dim=1)[0][:, None, None, None]
        elif self.norm == "L2":
            t = torch.randn_like(x)
            x_adv = x + self.eps * t / ((t**2).sum(dim=(1,2,3), keepdim=True).sqrt() + 1e-12)
        
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        
        loss_steps = torch.zeros(self.steps, x.shape[0])
        loss_best_steps = torch.zeros(self.steps+1, x.shape[0])
        acc_steps = torch.zeros_like(loss_best_steps)

        # 初始化最佳损失
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x_adv)
            text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
            similarity = (image_features_norm * text_features_norm).sum(dim=1)
            loss_best = similarity.clone() if not self.targeted else -similarity.clone()
        
        # 梯度初始化
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            x_adv.requires_grad_(True)
            image_features = self.clip_model.encode_image(x_adv)
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
            similarity = (image_features_norm * text_features_norm).sum(dim=1)
            
            if self.targeted:
                loss = -similarity.sum()
            else:
                loss = similarity.sum()
                
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        
        grad /= self.eot_iter
        grad_best = grad.clone()

        # 主循环
        step_size = self.eps * torch.ones_like(x) * 2
        x_adv_old = x_adv.clone()
        loss_best_last_check = loss_best.clone()
        
        for i in range(self.steps):
            # 梯度更新
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(x_adv + (x_adv_1 - x_adv)*0.75 + grad2*0.25, 0.0, 1.0)
                
                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * grad / (grad.pow(2).sum(dim=(1,2,3), keepdim=True).sqrt()+1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x).renorm(p=2, dim=0, maxnorm=self.eps), 0.0, 1.0)
                
                x_adv = x_adv_1

            # 计算梯度
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                x_adv.requires_grad_(True)
                image_features = self.clip_model.encode_image(x_adv)
                image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
                similarity = (image_features_norm * text_features_norm).sum(dim=1)
                
                if self.targeted:
                    loss = -similarity.sum()
                else:
                    loss = similarity.sum()
                    
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            
            grad /= self.eot_iter

            # 更新最佳样本
            with torch.no_grad():
                similarity = (image_features_norm * text_features_norm).sum(dim=1)
                curr_loss = similarity if not self.targeted else -similarity
                improve = curr_loss < loss_best
                x_best[improve] = x_adv[improve].clone()
                loss_best[improve] = curr_loss[improve]
                
                # 动态调整步长
                if i % 5 == 0:
                    fl_osc = self.check_oscillation(loss_steps.cpu().numpy(), i, 5)
                    fl_reduce = (loss_best_last_check.cpu() >= loss_best.cpu()).numpy()
                    fl_osc = np.logical_and(fl_osc, fl_reduce)
                    step_size[fl_osc] /= 2.0

                loss_best_last_check = loss_best.clone()
        
        return x_best, None, loss_best, x_best_adv

    def perturb(self, x, text_features, best_loss=False, cheap=True):
        # 在归一化空间中进行扰动
        adv = x.clone()
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
            similarity = (image_features_norm * text_features_norm).sum(dim=1)
            acc = similarity > 0.15  # 假设相似度阈值0.5为分类边界
        
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if ind_to_fool.numel() == 0:
                break
                
            x_to_fool = x[ind_to_fool].clone()
            text_to_fool = text_features[ind_to_fool].clone()
            
            _, _, _, adv_curr = self.attack_single_run(x_to_fool, text_to_fool)
            
            # 更新对抗样本
            with torch.no_grad():
                adv_features = self.clip_model.encode_image(adv_curr)
                adv_features_norm = adv_features / adv_features.norm(dim=1, keepdim=True)
                new_sim = (adv_features_norm * text_to_fool).sum(dim=1)
                success = new_sim < 0.5 if not self.targeted else new_sim > 0.5
                acc[ind_to_fool[success]] = False
                adv[ind_to_fool[success]] = adv_curr[success]
                
            if self.verbose:
                print(f"Restart {counter}: Success rate {1-acc.float().mean():.2%}")
        
        return adv



