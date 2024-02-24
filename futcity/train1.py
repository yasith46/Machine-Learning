#Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib


#Loading data and removing high cardinality categorical columns
print('Loading data...')
data = pd.read_csv('futcity/futuristic_city_traffic.csv')
y = data['Traffic Density']
X = data.drop(['Traffic Density', 'Energy Consumption'], axis=1)  # Check for improvements after removing economic status

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2)

cat_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == 'object']
num_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

cols = cat_cols + num_cols
X_train = X_train_full[cols].copy()
X_valid = X_valid_full[cols].copy()

print('Loaded data')
print('Processing...')

#Preprocessing for numerical data
num_trans = SimpleImputer(strategy = 'constant')

#Preprocessing for categorical data
cat_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_trans, num_cols),
        ('cat', cat_trans, cat_cols)
    ]
)

#Creating model
model = RandomForestRegressor(n_estimators = 100, random_state = 0)


#Evaluating
my_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', model)
])


my_pipeline.fit(X_train, y_train)
#print('Predicting...')
#preds = my_pipeline.predict(X_valid)

#score = mean_absolute_percentage_error(y_valid, preds)
#print('MAE: ', score)

joblib.dump(model, 'trained_model1.joblib')