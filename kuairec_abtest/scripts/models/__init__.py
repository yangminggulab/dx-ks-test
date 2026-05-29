"""
models/ — 推荐模型包

导出所有模型类和数据加载工具。

用法:
    from models import SASRec, BERT4Rec, SideInfoRec, CL4SRec, TwoTower
    from models.kuairec_loader import load_model_data
    from models.base import ModelData, BaseRecommender
"""

from models.sasrec    import SASRec
from models.bert4rec  import BERT4Rec
from models.sideinfo  import SideInfoRec
from models.cl4srec   import CL4SRec
from models.two_tower import TwoTower

__all__ = ["SASRec", "BERT4Rec", "SideInfoRec", "CL4SRec", "TwoTower"]
