import torch
import torch.nn as nn


class Heads(nn.Module):
    """Heads for tricks classification and regression

    Parameters
    ----------
    in_features : int
        Number of features generated from prior layers

    byrt : list
        A list of size hidden layers with number of neurons per layer
        to predict body rotation type

    byrn : list
        A list of size hidden layers with number of neurons per layer
        to predict body rotation number

    bdrt : list
        A list of size hidden layers with number of neurons per layer
        to predict board rotation type

    bdrn : list
        A list of size hidden layers with number of neurons per layer
        to predict board rotation number

    ft : list
        A list of size hidden layers with number of neurons per layer
        to predict flip type

    fn : list
        A list of size hidden layers with number of neurons per layer
        to predict number of flips

    landed : list
        A list of size hidden layers with number of neurons per layer
        to predict whether or not trick was landed

    stance : list
        A list of size hidden layers with number of neurons per layer
        to predict stance
    """
    def __init__(
        self,
        in_features: int,
        byrt: list[tuple[int, float]],
        byrn: list[tuple[int, float]],
        bdrt: list[tuple[int, float]],
        bdrn: list[tuple[int, float]],
        ft: list[tuple[int, float]],
        fn: list[tuple[int, float]],
        landed: list[tuple[int, float]],
        stance: list[tuple[int, float]]
    ) -> None:
        super(Heads, self).__init__()
        self.in_features = in_features
        self.byrt_net = self.build_head(byrt, 3)
        self.byrn_net = self.build_head(byrn, 1)
        self.bdrt_net = self.build_head(bdrt, 3)
        self.bdrn_net = self.build_head(bdrn, 1)
        self.ft_net = self.build_head(ft, 3)
        self.fn_net = self.build_head(fn, 1)
        self.landed_net = self.build_head(landed, 2)
        self.stance_net = self.build_head(stance, 4)

    def build_head(self, neurons_list: list, features: int) -> nn.Sequential:
        net = nn.Sequential()
        net.append(nn.Linear(self.in_features, neurons_list[0][0]))
        net.append(nn.ReLU())
        for idx, (neurons, dropout) in enumerate(neurons_list):
            if idx==len(neurons_list)-1:
                net.append(nn.Dropout(dropout))
                net.append(nn.Linear(neurons, features))
                continue
            net.append(nn.Dropout(dropout))
            net.append(nn.Linear(neurons, neurons_list[idx+1][0]))
            net.append(nn.ReLU())
        
        return net

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        out["body_rotation_type"] = self.byrt_net(x)
        out["body_rotation_number"] = self.byrn_net(x)
        out["board_rotation_type"] = self.bdrt_net(x)
        out["board_rotation_number"] = self.bdrn_net(x)
        out["flip_type"] = self.ft_net(x)
        out["flip_number"] = self.fn_net(x)
        out["landed"] = self.landed_net(x)
        out["stance"] = self.stance_net(x)

        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x   