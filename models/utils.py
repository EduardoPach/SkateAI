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
        byrt: list,
        byrn: list,
        bdrt: list,
        bdrn: list,
        ft: list,
        fn: list,
        landed: list,
        stance: list
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
        net.append(nn.Linear(self.in_features, neurons))
        net.append(nn.ReLU())
        for idx, neurons in enumerate(neurons_list):
            if idx==len(neurons_list):
                net.append(nn.Linear(neurons, features))
                continue
            net.append(nn.Linear(neurons, neurons_list[idx+1]))
            net.append(nn.ReLU())
        
        return net

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        byrt_out = self.byrt_net(x)
        byrn_out = self.byrn_net(x)
        bdrt_out = self.bdrt_net(x)
        bdrn_out = self.bdrn_net(x)
        ft_out = self.ft_net(x)
        fn_out = self.fn_net(x)
        landed_out = self.landed_net(x)
        stance_out = self.stance_net(x)

        return [byrt_out, byrn_out, bdrt_out, bdrn_out, ft_out, fn_out, landed_out, stance_out]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x   