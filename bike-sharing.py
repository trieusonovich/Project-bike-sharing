import pandas as pd
from sklearn.compose import ColumnTransformer
#from ydata_profiling import ProfileReport
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, \
    QuantileTransformer, OneHotEncoder
from feature_engine.creation import CyclicalFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('bike-sharing.csv', header = 0,sep = ',')
# profile = ProfileReport(data, title = 'report_bike_sharing')
# profile.to_file('report_bike_sharing.html')
#date = pd.to_datetime(data['date_time'],yearfirst=True)

x = data.drop(['date_time','users'],axis=1)
y = data['users']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle = False)

num_quan_trans = Pipeline(steps=[('quantile',QuantileTransformer(n_quantiles=1700,output_distribution='normal')),
                                 ('scaler', StandardScaler())])
num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                            ('scaler', MinMaxScaler(feature_range=(0,1)))])
num_bin_trans = Pipeline(steps=[('bins', KBinsDiscretizer(n_bins=3,encode = 'ordinal',strategy='quantile'))])
nom_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
time_trans = Pipeline(steps=[('cyclical', CyclicalFeatures(
                          variables=['month','hour', 'weekday'],
                          drop_original=True,
                          max_values={"month": 12,"hour": 24, "weekday": 7}))])
# transformed_data = time_trans.fit_transform(data[['month','hour', 'weekday']])
# print(transformed_data)

preprocessor = ColumnTransformer(transformers=[('num_quan_trans', num_quan_trans, ['atemp']),
                                               ('num_trans', num_trans, ['temp','hum']),
                                               ('num_bin_trans', num_bin_trans, ['windspeed']),
                                               ('nom_trans', nom_trans,['weather']),
                                               ('time_trans',time_trans,['month','hour', 'weekday'])
], remainder='passthrough')

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor',RandomForestRegressor())])

params = {
    "regressor__n_estimators": [100, 500],
    "regressor__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "regressor__max_depth": [None,5,30],
    'regressor__min_samples_split': [10,30],
    'regressor__min_samples_leaf': [20,40],
    'regressor__max_features': ['sqrt', 'log2', None]
    # "preprocessor__num_quan_trans__quantile__output_distribution": ["uniform", "normal"],
    # "preprocessor__num_trans__scaler": ["MinMaxScaler", "StandardScaler"],
    # "preprocessor__num_bin_trans__bins__strategy": ["quantile", "uniform"]
}

gs_model = GridSearchCV(
    estimator= model,
    param_grid= params,
    scoring= "r2",
    cv = 4,
    verbose=1,
    n_jobs= 6
)
gs_model.fit(x_train, y_train)
print(gs_model.best_params_)
print(gs_model.best_score_)
best_model = gs_model.best_estimator_

y_predict = best_model.predict(x_test)
print('MSE:{}'.format(mean_squared_error(y_test,y_predict)))
print('MAE:{}'.format(mean_absolute_error(y_test,y_predict)))
print('R2:{}'.format(r2_score(y_test,y_predict)))

size = int(len(x)*0.8)
fig,ax = plt.subplots()
ax.plot(data['date_time'][:size], data['users'][:size],label='train')
ax.plot(data['date_time'][size:], data['users'][size:],label='test')
ax.plot(data['date_time'][size:], y_predict,label='predict')
