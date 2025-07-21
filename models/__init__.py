# from .pointnet2_sem_seg import get_model as get_pointnet2
# from .pct_seg import get_model as get_pct
# # from .Sennet import KPFCNN
# from .pt_hmamba import get_model as get_ptmamba
# from .pcm_seg import get_model as get_pcm
# model_factory = {
#     "pointnet2": get_pointnet2,
#     "pct": get_pct,
#     "pt_mamba": get_ptmamba,
#     "pcm": get_pcm
# }




# def get_model(model_name, **kwargs):
#     if model_name not in model_factory:
#         raise ValueError(f"Unknown model name: {model_name}")
#     return model_factory[model_name](**kwargs)



from .pointnet2_pl import PointNet2_pl
from .pointnet_pl import PointNet_pl

model_dict = {
    'pointnet2': PointNet2_pl,
    'pointnet': PointNet_pl,
}

def build_model(config):
    model_type = config.model.model_type
    model_class = model_dict[model_type]
    return model_class(config)
