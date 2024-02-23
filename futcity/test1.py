#Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#Loading data and removing high cardinality categorical columns
data = pd.read_csv('futuristic_city_traffic.csv')
y = data['Traffic Density']
X = data.drop(['Traffic Density', 'Energy Consumption'], axis=1)  # Check for improvements after removing economic status

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3)

cat_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == 'object']
num_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

cols = cat_cols + num_cols
X_train = X_train_full[cols].copy()
X_valid = X_valid_full[cols].copy()


X_train.head()