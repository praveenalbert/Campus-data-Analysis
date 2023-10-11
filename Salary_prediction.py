import pandas as pd
from  sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('all_tiers_data_2.csv')

df.drop(columns=['s_id','name','profile_link','gender'],inplace=True)


df['inter_gpa'] = df['inter_gpa'].fillna(df['inter_gpa'].mean())

df['ssc_gpa'] = df['ssc_gpa'].fillna(df['ssc_gpa'].mean())

df['cgpa'] = df['cgpa'].fillna(df['cgpa'].mean())

dummy = pd.get_dummies(df.branch)
df=pd.concat([df,dummy],axis=1)
df.drop(columns=['EEE','branch','other_skills','ssc_gpa','inter_gpa'],inplace=True)

X = df.drop(columns=['salary_as_fresher','is_placed'],axis=1)
y=df['salary_as_fresher']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
# Define the hyperparameters and their respective search spaces
param_grid = {
    'n_estimators': [50, 100],   # Number of trees in the forest
    'max_depth': [None, 10, 20],      # Maximum depth of the trees
    'min_samples_split': [2, 5, 10], # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]    # Minimum samples required to be at a leaf node
}
# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Use GridSearchCV to perform a grid search over the hyperparameter space
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)
best_params = grid_search.best_params_
# Instantiate a new RandomForestRegressor with the best hyperparameters
model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                max_depth=5,  # Try a smaller value
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'],
                                random_state=42)


# Train the model with the entire dataset
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
import pickle
pickle.dump(model,open('salary_model.pkl','wb'))