import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df.head()

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sex = pd.get_dummies(df['Gender'],drop_first=True)
married = pd.get_dummies(df['Married'],drop_first=True)
education = pd.get_dummies(df['Education'],drop_first=True)
self_emp = pd.get_dummies(df['Self_Employed'],drop_first=True)
loanStatus = pd.get_dummies(df['Loan_Status'],drop_first=True)
PropArea = pd.get_dummies(df['Property_Area'],drop_first=True)
df.drop(['Gender','Loan_ID','Self_Employed','Married','Education','Property_Area'],axis=1,inplace=True)
df = pd.concat([df,married,education,sex,self_emp,loanStatus,PropArea],axis=1)
df.head()

df=df.rename(columns={'Yes':'Married','Male':'Gender','Yes':'Self_Employed'})


def impute_LoanAmt(cols):
    Loan = cols[0]
    selfemp = cols[1]

    if pd.isnull(Loan):

        if selfemp == 1:
            return 150
        else:
            return 125

    else:
        return Loan

df['LoanAmount'] = df[['LoanAmount','Self_Employed']].apply(impute_LoanAmt,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.countplot(x='Credit_History',data=df,palette='RdBu_r')

df['Credit_History'].fillna(1.0,inplace=True)

sns.countplot(x='Loan_Amount_Term',data=df,palette='RdBu_r')

df['Loan_Amount_Term'].fillna(360,inplace=True)

sns.countplot(x='Dependents',data=df,palette='RdBu_r')

df['Dependents'].fillna(360,inplace=True)

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(df.columns)
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
scaler = StandardScaler()
train=pd.DataFrame(df.drop('Loan_Status',axis=1))
scaler.fit(train)
scaled_features = scaler.transform(train)
df_feat = pd.DataFrame(scaled_features,columns=train.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Loan_Status'],test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))

x=df[[ 'Not Graduate', 'Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','Semiurban','Urban','Loan_Amount_Term']]
y=df['LoanAmount']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression, BayesianRidge

lm=LinearRegression()
lm.fit(X_train,Y_train)

print(lm.intercept_)

coeff=pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff

lm.predict([[1,0,3,1,4000,3000,0,1,360]])

from sklearn import linear_model

bm=linear_model.BayesianRidge()
bm.fit(X_train,Y_train)

BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
              compute_score=False, copy_X=True, fit_intercept=True,
              lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300,
              normalize=False, tol=0.001, verbose=False)

bm.predict([[1,0,3,1,4000,3000,0,1,360,1.0]])

from xgboost import XGBRegressor
model=XGBRegressor()
model.fit(X_train,Y_train)

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

yhat = model.predict([[1,0,3,1,4000,3000,0,1,360,1.0]])

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train,Y_train)

RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=2, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=0, verbose=0, warm_start=False)

regr.predict([[1,0,3,1,4000,3000,0,1,360,1.0]])