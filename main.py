import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ai_models import har_model as model
from internal import util, math, plot

plt.style.use('ggplot')
dataset = util.read_data('WISDM_at_v2.0_raw.txt', 1)
dataset['x-axis'] = math.normalize(dataset['x-axis'])
dataset['y-axis'] = math.normalize(dataset['y-axis'])
dataset['z-axis'] = math.normalize(dataset['z-axis'])

unique_activities = pd.unique(dataset["activity"])
for activity in unique_activities:
    subset = dataset[dataset["activity"] == activity][:180]
    plot.activity(activity, subset)

segments, labels = util.create_segments(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

train_test_split = np.random.rand(len(reshaped_segments)) < 0.50
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

input_height = train_x.shape[1]
input_width = train_x.shape[2]
num_channels = train_x.shape[3]
filter_width = 60
channels_multiplier = 60
num_hidden = 1000
filter_width_2 = 6
num_labels = unique_activities.shape[0]
model_instance = model(num_labels, input_height, input_width, num_channels, filter_width,
                         channels_multiplier,
                         num_hidden, filter_width_2)
model_instance.train_and_evaluate(train_x, train_y, test_x, test_y)
