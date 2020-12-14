import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = 'train.csv'
pd_data = pd.read_csv(data)
pd_data = pd_data.dropna(axis=1)
features = ["LotArea", "OverallQual" , "OverallCond", 
                   "YearBuilt", "YearRemodAdd"]
y = pd_data['SalePrice']
X = pd_data[features]
# DesicionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(X, y)
tree_pred = tree_model.predict(X)
#print(tree_pred[:5])
#print(y.head())

#Let's split using train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
def g_vals():
    print(sp_pred[:10])
    print(val_y.head(10))
    mae = mean_absolute_error(val_y, sp_pred)
    print(mae)
sp_tr_model = DecisionTreeRegressor(random_state=0)
sp_tr_model.fit(train_X, train_y)

sp_pred = sp_tr_model.predict(val_X)
#g_vals()

# RandomForestRegressor
# max_leaf_nodes is unlimited =none
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(train_X, train_y)
sp_pred = rf_model.predict(val_X)
#g_vals()

#Get the best_leaf_node

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    rf_model_op = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, 
    random_state=0)
    rf_model_op.fit(train_X, train_y)
    pred = rf_model_op.predict(val_X)
    return mean_absolute_error(val_y, pred)

# Getting the best node
"""nodes = [i for i in range(41, 69)]
err, best_nodes = get_mae(nodes[0], train_X, val_X, train_y, val_y), nodes[0]
for node in nodes:
    if (mae := get_mae(node, train_X, val_X, train_y, val_y)) < err:
        err, best_nodes = mae, node"""
    #print(f"Mean absolute error for leaf_nodes {node} is {get_mae(node, train_X, val_X, train_y, val_y)}")
#print(best_nodes)
best_nodes = 55
#Finally for the whole data set
model = RandomForestRegressor(max_leaf_nodes=best_nodes, random_state=0)
model.fit(X, y)

#test value

test = 'test.csv'
pd_test = pd.read_csv(test)
pd_test = pd_test.dropna(axis=1)
test_X = pd_test[features]

predictions = model.predict(test_X)
print(predictions)
#output = pd.DataFrame({'id': pd_test.Id,
#                                        'SalePrice': predictions})
#output.to_csv("submission.csv", index=False)