import colorama

from ..api import Path
from ..api.utils import pose_tf

colorama.init(autoreset=True)


def process_config(config: dict,
                   with_api: bool = True) -> dict:
    if not isinstance(config, dict): config = Path(config).yaml()
    pose_tf.TFmanager.load_yaml(config["transform"])

    if with_api:
        try:
            register_tog_model(config["tog_model"])
        except ImportError as e:
            print(colorama.Fore.RED + e)

    return config


def register_tog_model(config: dict):
    from .utils.model_wrapper import TOGEvaluatorAndAffordance
    from .api import FUNC_API
    ret = FUNC_API.functions["tog"] = TOGEvaluatorAndAffordance(config)
    return ret
