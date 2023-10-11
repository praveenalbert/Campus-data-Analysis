from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('all_tiers_data_2.csv')
df.drop(columns=['s_id','profile_link','name', 'gender', 'salary_as_fresher'], inplace=True)
df['inter_gpa'] = df['inter_gpa'].fillna(df['inter_gpa'].mean())

# Fill missing values in 'ssc_gpa' column with the mean of non-missing values
# No missing values in 'ssc_gpa', so this step doesn't change the column.

df['ssc_gpa'] = df['ssc_gpa'].fillna(df['ssc_gpa'].mean())

# Fill missing values in 'cgpa' column with the mean of non-missing values
# This replaces the missing value with the mean of the available 'cgpa' values.

df['cgpa'] = df['cgpa'].fillna(df['cgpa'].mean())

#one hot coding
dummy = pd.get_dummies(df.branch)
df = pd.concat([df, dummy], axis=1)
df.drop(columns=['EEE', 'branch'], inplace=True)
df.drop(columns=['other_skills'],inplace=True)
df['is_placed']=df['is_placed'].astype(int)
# Loading the Independent features into X variable
X=df.drop(columns=['is_placed'])
y=df['is_placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=5)
model=LogisticRegression()
model.fit(X_train,y_train)
output = model.predict(X_test)
print(accuracy_score(output,y_test))
print(output)
import pickle
pickle.dump(model,open('Placement_model.pkl','wb'))