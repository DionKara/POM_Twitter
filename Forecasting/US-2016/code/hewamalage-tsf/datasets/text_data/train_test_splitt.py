import pandas as pd

data = pd.read_csv("Rep_time_series_MEAN.csv")


train_data = data.iloc[:,1:157] # change the train portion for each case
train_data['Descriptor'] = data['Descriptor']

cols = train_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
train_data = train_data[cols]

test_data = data.iloc[:,157:164] # change the test portion for each case
test_data['Descriptor'] = data['Descriptor']

cols = test_data.columns.tolist()
cols = cols[-1:] + cols[:-1]
test_data = test_data[cols]

train_data.to_csv("time_series_train.csv", index=False)

test_data.to_csv("time_series_test.csv", index=False)