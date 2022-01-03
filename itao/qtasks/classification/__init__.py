from itao.qtasks.classification.train import TrainCMD
from itao.qtasks.classification.retrain import ReTrainCMD
from itao.qtasks.classification.eval import EvalCMD
from itao.qtasks.classification.prune import PruneCMD
from itao.qtasks.classification.export import ExportCMD
from itao.qtasks.classification.inference import InferCMD

__all__ = [
    'TrainCMD',
    'ReTrainCMD',
    'EvalCMD',
    'PruneCMD',
    'ExportCMD',
    'InferCMD'
]