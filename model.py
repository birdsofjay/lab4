from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import pickle

data = '''back dos
buffer_overflow u2r
ftp_write r2l
guess_passwd r2l
imap r2l
ipsweep probe
land dos
loadmodule u2r
multihop r2l
neptune dos
nmap probe
perl u2r
phf r2l
pod dos
portsweep probe
rootkit u2r
satan probe
smurf dos
spy r2l
teardrop dos
warezclient r2l
warezmaster r2l'''

# grouped by type
attack_types = pd.DataFrame([row.split() for row in data.split('\n')], columns=['name','type'])
attack_type_groups = attack_types.groupby('type')['name'].unique()

print('attack group types: {}'.format(', '.join(attack_type_groups.index)))
print()
print(attack_type_groups)

#X = features
#y = label (target)
from sklearn.datasets import fetch_openml
X, y = fetch_openml(data_id='1113', return_X_y=True, as_frame=True)
print('n records: {}'.format(len(X.index)))
X_preserved = X.copy()
y_preserved = y.copy()

def get_attack_type_downsampled_balanced_subset(attack_names, label, X, y):
    print('Attack group name: {}'.format(label))
    print('Attack_types: {}'.format(', '.join(attack_names)))
    
    is_type_attack = y.isin(attack_names)
    
    only_attack_type = y[is_type_attack]
    only_not_attack_type = y[~is_type_attack]
    
    only_attack_type = is_type_attack[is_type_attack]
    only_not_attack_type = is_type_attack[~is_type_attack]
    
    
    num_attack_type = only_attack_type.shape[0]
    num_not_attack_type = only_not_attack_type.shape[0]
    
    print('Num attack type: {}'.format(num_attack_type))
    print('Num not attack type: {}'.format(num_not_attack_type))
    

    # Take a balanced sample
    # which one has less? that is the one we should downsample
    lowest_count = min(num_attack_type, num_not_attack_type)
    
    balanced_ys = []
    balanced_Xs = []
    for subset_y in [only_attack_type, only_not_attack_type]:
        _subset_y = subset_y.copy()
        if _subset_y.shape[0] > lowest_count:
            _subset_y = subset_y.sample(n=lowest_count)
        subset_X = X.loc[_subset_y.index, :]
        balanced_Xs.append(subset_X)
        balanced_ys.append(_subset_y)
    
    assert len(balanced_Xs) == len(balanced_ys)
    
    for i, balanced_y in enumerate(balanced_ys):
        assert balanced_y.shape[0] == lowest_count
        assert balanced_Xs[i].shape[0] == lowest_count
        
    X_new = pd.concat(balanced_Xs)
    y_new = pd.concat(balanced_ys).rename(label)
    
    print(X_new.shape[0])
    print(y_new.shape[0])
    print()
    
    return X_new, y_new

X_is_dos, y_is_dos = get_attack_type_downsampled_balanced_subset(attack_type_groups['dos'], 'is_dos_attack', X, y)
X_is_probe, y_is_probe = get_attack_type_downsampled_balanced_subset(attack_type_groups['probe'], 'is_probe_attack', X, y)
X_is_r2l, y_is_r2l = get_attack_type_downsampled_balanced_subset(attack_type_groups['r2l'], 'is_r2l_attack', X, y)
X_is_u2r, y_is_u2r = get_attack_type_downsampled_balanced_subset(attack_type_groups['u2r'], 'is_u2r_attack', X, y)

X, y = X_is_probe, y_is_probe

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


np.random.seed(0)

#column transformer

numeric_features = ['src_bytes','dst_bytes']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['protocol_type']
#categorical_features = []
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

classifiers = [
    LogisticRegression()
]

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('clf', None)])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

roc_things = []
precision_recall_things = []

for classifier in classifiers:
    clf.set_params(clf=classifier).fit(X_train, y_train)
    classifier_name = classifier.__class__.__name__
    print(str(classifier))
    print("model score: %.3f" % clf.score(X_test, y_test))

    y_score = clf.predict_proba(X_test)[:,1]

    y_pred = clf.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_things.append((fpr, tpr, '{} AUC: {:.3f}'.format(classifier_name, roc_auc)))
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    precision_recall_things.append((recall, precision, thresholds, '{} AUC: {:.3f}'.format(classifier_name, pr_auc)))
    #plot_precision_recall_curve(clf, X_test, y_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('average precision score: {:.3f}'.format(average_precision_score(y_test, y_score)))
    print('roc_auc_score: {:.3f}'.format(roc_auc))
    print('precision-recall AUC: {:.3f}'.format(pr_auc))
    print()

roc_plt = plt.figure()
lw = 4
for roc_thing in roc_things:
    fpr, tpr, label = roc_thing
    plt.plot(fpr, tpr, lw=lw, label=label)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.legend()
plt.title('ROC curve')

pr_plt = plt.figure()
for pr_thing in precision_recall_things:
    recall, precision, _, label = pr_thing
    plt.plot(recall, precision, lw=lw, label=label)
ratio = y_test[y_test].shape[0] / y_test.shape[0]
plt.hlines(y=ratio, xmin=0, xmax=1, color='navy', lw=lw, linestyle='--')
plt.title('Precision-recall plot')
plt.legend()

with open('{}.pkl'.format(classifier_name), 'wb') as f:
    pickle.dump(clf, f)