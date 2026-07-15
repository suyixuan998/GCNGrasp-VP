import colorama

from ..api.utils.func_invoker import LocalInvoker

# functions
FUNCTIONS = {}

try:
    from ..api.contact_graspnet import ContactGraspNet
    from ..api.grounded_sam import GroundedSAM

    cgn = ContactGraspNet()
    gsam = GroundedSAM(gdino_kwargs=dict(box_thresh=.2), sam_kwargs=dict(encoder="large"))

    FUNCTIONS.update({
        "cgn": cgn,
        "cgn_frame": cgn.process_frame,
        "gsam": gsam,
    })
except ImportError as e:
    print(colorama.Fore.RED + e)

FUNC_API = LocalInvoker(FUNCTIONS)


def get_da3_mapper(enable_main: bool = True,
                   enable_mono: bool = True):
    try:
        from gcngrasp.api.depth_anything import DA3Mapper

        mapper = DA3Mapper(size="g" if enable_main else None, enable_mono=enable_mono, align_mode="last")
        return lambda x: mapper(x).frames_rel

    except ImportError as e:
        print(e)
        return
