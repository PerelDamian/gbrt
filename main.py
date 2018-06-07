from data_utils import parse_data
from gbrt_algorithm import gbrt

train_dataset, test_dataset = parse_data('data/train.csv')
model = gbrt(train_data=train_dataset.data, test_data=test_dataset.data, num_trees=5,
             max_depth=3, min_node_size=1, label_name=train_dataset.label_name)
