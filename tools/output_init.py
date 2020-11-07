from .output_tool import basic_output_function, null_output_function, ConsGraphOutputFunc, RankOutputFunc, BinaryOutputFunc
from .output_tool import BinaryOutputFunc2
output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "ConsGraph": ConsGraphOutputFunc,
    "Rank": RankOutputFunc,
    "binary": BinaryOutputFunc,
    "binary2": BinaryOutputFunc2,
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
