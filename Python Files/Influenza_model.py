import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
train =pd.read_csv(r'C:\Users\ArtisusXiren\Desktop\Reinforcemtn\large_data.csv')
Mapping={value:num for num, value in enumerate(train['TYPE'].unique())}
train['TYPE']=train['TYPE'].map(Mapping)
null=train.isnull().sum()
attributes=[columns for columns in train if columns!='TYPE']
X=train[attributes].values
y=train['TYPE'].values
f_2,z=f_classif(X,y)
k_best=SelectKBest(score_func=f_classif,k=12)
x_new=k_best.fit_transform(X,y)
index=k_best.get_support(indices=True)
results=[attributes[i] for i in index]
features_to_train=pd.DataFrame(x_new,columns=results)
features_to_train['TYPE']=train['TYPE']
X_train, X_test, y_train, y_test= train_test_split(x_new,y,test_size=0.2,random_state=42)
Model= SVC(kernel='linear',random_state=42)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(accuracy_score(y_test,y_pred))
joblib.dump(Model,'Model.pkl')

