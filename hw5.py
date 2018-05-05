'''Defining Imports'''
from pandas import *
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import operator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import statistics
from sklearn.utils import resample
from sklearn.feature_selection import RFE



# print('Young People Survey Dataset - Task 2\n\n')

'''Pulling the data from the CSV files'''
#Loading data
print('Fetching required data...\n\n')
fname = 'responses.csv'
f = open(fname, 'r') 

df = pandas.read_csv(fname)
# print(df.describe())
# print(len(df.columns))



# X_sparse = resample(df(df['Spending on healthy eating'] == 1), n_samples = 70, random_state=10, replace=True)
X_sparse = df[df['Spending on healthy eating'] == 1]
# X_sparse

X_resampled = resample(X_sparse, n_samples = 70, replace = True, random_state = 70)
# print(X_resampled.shape)
# X_resampled
df = concat([df, X_resampled], axis = 0)
# df.shape

# df = new_df
# df = df + X_resampled
# df

# print(df.shape)
print('Preprocessing data...\n')
# Fill mean for numerical cols        
for each_df in df:
#     print(X[each_x].dtypes)

    if df[each_df].dtypes != 'object' and each_df != 'Spending on healthy eating':
        df[each_df].fillna(df[each_df].mean(), inplace=True)
# print('\n\n\n\n')        
# print(df.isnull().sum())

# Fill mode for the categorical data cols:
for each_df1 in df:
    if df[each_df1].dtypes == 'object':
        df[each_df1].fillna(statistics.mode(df[each_df1]), inplace = True)
# print('\n\n\n\n')        
# print(df.isnull().sum()) 
print('Dropping rows with no label values in the target column...\n')
df.dropna(inplace = True)
# df.describe()
df.reset_index(drop = True, inplace = True)
print('Dataset now contains %d rows and %d columns...\n' %(df.shape[0], df.shape[1]))


print('Preparing input features and output labels...')
X = df.drop(['Spending on healthy eating'], axis = 1)
# print(X.shape)
Y = df['Spending on healthy eating']
# print(X.shape, Y.shape)

column_types = X.columns
# print(column_types)
le = LabelEncoder()
for index, col in enumerate(column_types):
    
    if X[col].dtypes == 'object':
        le.fit(X[col])
#         print(le.classes_)
        X[col] = le.transform(X[col])
#         print(X[col])
#         print('*********')

# print(X.isnull().sum().sum())
X = X.round()
Y = Y.round()



# Fill mean for numerical cols        
for each_df in df:
#     print(X[each_x].dtypes)

    if df[each_df].dtypes != 'object' and each_df != 'Spending on healthy eating':
        df[each_df].fillna(df[each_df].mean(), inplace=True)
# print('\n\n\n\n')        
# print(df.isnull().sum())



# Fill mode for the categorical data cols:
for each_df1 in df:
    if df[each_df1].dtypes == 'object':
        df[each_df1].fillna(statistics.mode(df[each_df1]), inplace = True)
# print('\n\n\n\n')        
# print(df.isnull().sum())        

df.dropna(inplace = True)
# df.describe()
df.reset_index(drop = True, inplace = True)

X = df.drop(['Spending on healthy eating'], axis = 1)
# print(X.shape)
Y = df['Spending on healthy eating']
# print(X.shape, Y.shape)


column_types = X.columns
# column_types


le = LabelEncoder()
for index, col in enumerate(column_types):
    
    if X[col].dtypes == 'object':
        le.fit(X[col])
        print(le.classes_)
        X[col] = le.transform(X[col])
#         print(X[col])
#         print('*********')
print('Checking for any empty values in the feature set...')
print(X.isnull().sum().sum())

X = X.round()
Y = Y.round()
# X

print('*********************************************************************************************\n')

print('*** BASELINE MODEL ***\n')
print('Using Logistic Regression, not selecting any features...\n')


log_reg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X.round(), Y.round(), test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train.round(), Y_train.round(), test_size = 0.1)
log_reg.fit(X_train, Y_train)
log_reg_pred = log_reg.predict(X_val)
print('Validation Accuracy')
print(np.mean(log_reg_pred == Y_val))

log_reg_pred = log_reg.predict(X_test)
print('Test Accuracy')
print(np.mean(log_reg_pred == Y_test))

print('*********************************************************************************************\n')

print('\n\n*** Feature Selection and Applying ML Models ***\n')
'''Feature Selection Techniques'''
'''Finding correlation by Numpy's corrcoef '''

corr_arr = {}
for each_x in X:
    corr_arr[each_x] = np.corrcoef(x=X[each_x], y=Y)[0,1]
# print(corr_arr)
sorted_corr_arr = sorted(corr_arr.items(), key=operator.itemgetter(1), reverse=True)
# type(sorted_corr_arr)
X_corr = sorted_corr_arr[:30]
new_X = []
for row in range(0, len(X_corr)):
    new_X.append(X_corr[row][0])
# new_X

X_corr = X[new_X]
# X_corr

print('*********************************************************************************************\n')

print('*** PROPOSED MODEL ***')
print('\nFinding & selecting highly correlated features and applying SVC (30 features)...\n')


'''Splitting data --- Top 30 correlated features -- SVC'''


X_train, X_test, Y_train, Y_test = train_test_split(X_corr.round(), Y.round(), test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train.round(), Y_train.round(), test_size = 0.1)

svc_clf = SVC(C=2, kernel = 'rbf')
svc_clf.fit(X_train, Y_train)
l_svc_pred_val = svc_clf.predict(X_val)
print('The accuracy of the model on: ')
print('Validation Data:'),
print(np.mean(l_svc_pred_val == Y_val))

print('***')

# svc_clf = SVC(multi_class = 'ovr', C=2)
svc_clf.fit(X_train, Y_train)
l_svc_pred = svc_clf.predict(X_test)
print('Test Data:')
print(np.mean(l_svc_pred == Y_test))
# print(l_svc_pred)
print('\n')

print('***')

print('Confusion Matrix for this model:')

target_names = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']
print(classification_report(Y_test, l_svc_pred, target_names=target_names))

print('*********************************************************************************************\n')

'''Splitting data --- Top 30 correlated features -- Random Forest'''
print('\nTop 30 correlated features -- Using Random Forest Model')



X_train, X_test, Y_train, Y_test = train_test_split(X_corr.round(), Y.round(), test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train.round(), Y_train.round(), test_size = 0.1)

randF_clf = RandomForestClassifier(n_estimators = 6, max_depth = 6, max_features = 30, criterion = 'gini', warm_start = False)
randF_clf.fit(X_train, Y_train)
l_randF_pred = randF_clf.predict(X_val)
print('Validation Accuracy')
print(np.mean(l_randF_pred == Y_val))

# randF_clf = RandomForestClassifier(max_features = 10, criterion = 'entropy', warm_start = False)
randF_clf.fit(X_train, Y_train)
l_randF_pred = randF_clf.predict(X_test)
print('Test Accuracy')
print(np.mean(l_randF_pred == Y_test))

print('*********************************************************************************************')

print('\nTop 30 correlated features -- Using Logistic Regression Model')


log_reg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X_corr.round(), Y.round(), test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train.round(), Y_train.round(), test_size = 0.1)
log_reg.fit(X_train, Y_train)
log_reg_pred = log_reg.predict(X_val)
print('The accuracy of the model on: ')
print('Validation Data:'),
print(np.mean(log_reg_pred == Y_val))

log_reg_pred = log_reg.predict(X_test)
print('Test Data: '),
print(np.mean(log_reg_pred == Y_test))

print('*********************************************************************************************')


print('Using Recursive Feature Elimination for selecting features and applying Logistic Regression...\n')
log_reg = LogisticRegression(multi_class='ovr', C = 2, solver='newton-cg', max_iter=105)

rfe = RFE(log_reg, 30)
print('Fitting model on 30 features...')
fit_ = rfe.fit(X_train, Y_train)
# print(fit_.support_)
pred = rfe.predict(X_val)
print('The accuracy of the model on: ')
print('Validation Data:'),
# print(np.mean(pred == Y_val))
score = accuracy_score(Y_val.round(), pred)
print(score)

pred = rfe.predict(X_test)
print('Test Data:'),
# print(np.mean(pred == Y_val))
score = accuracy_score(Y_test.round(), pred)
print(score)
print(classification_report(Y_test, pred, target_names=target_names))

print('~~ End ~~')