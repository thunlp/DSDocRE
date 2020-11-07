from .NaiveDataset import NaiveDataset
from .PreTrainDataset import PreTrainDataset

dataset_list = {
    "Naive": NaiveDataset,
    "PreTrain": PreTrainDataset,
}
