from .FineTuneBert import FineTuneBert
from .DSDocREBert import DSDocREBert
from .PreDenoiseBert import PreDenoiseBert

model_list = {
    "FineTune": FineTuneBert,
    "PreTrain": DSDocREBert,
    "PreDenoise": PreDenoiseBert
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
