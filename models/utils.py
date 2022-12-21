import torch
import torch.nn as nn


class Heads(nn.Module):
    """_summary_

    Parameters
    ----------
    in_features : int
        _description_
    brt : list
        _description_
    brn : list
        _description_
    ft : list
        _description_
    fn : list
        _description_
    landed : list
        _description_
    stance : list
        _description_
    """
    def __init__(
        self,
        in_features: int,
        brt: list,
        brn: list,
        ft: list,
        fn: list,
        landed: list,
        stance: list
    ) -> None:
        super(Heads, self).__init__()
        self.in_features = in_features
        self.brt_net = self.build_head(brt, 3)
        self.brn_net = self.build_head(brn, 1)
        self.ft_net = self.build_head(ft, 3)
        self.fn_net = self.build_head(fn, 1)
        self.landed_net = self.build_head(landed, 2)
        self.stance_net = self.build_head(stance, 4)

    def build_head(self, neurons_list: list, features: int) -> nn.Sequential:
        net = nn.Sequential()
        for idx, neurons in enumerate(neurons_list):
            if idx==0:
                net.append(nn.Linear(self.in_features, neurons))
                net.append(nn.ReLU())
            elif idx==len(neurons_list):
                net.append(nn.Linear(neurons, features))
            else:
                net.append(nn.Linear(neurons_list[idx-1], neurons))
                net.append(nn.ReLU())
        
        return net

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        brt_out = self.brt_net(x)
        brn_out = self.brn_net(x)
        ft_out = self.ft_net(x)
        fn_out = self.fn_net(x)
        landed_out = self.landed_net(x)
        stance_out = self.stance_net(x)

        return [brt_out, brn_out, ft_out, fn_out, landed_out, stance_out]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x   