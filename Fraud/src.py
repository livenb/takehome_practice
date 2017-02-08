import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler


def load_data(data_path):
    fraud_file = os.path.join(data_path, 'Fraud_Data.csv')
    ip_file = os.path.join(data_path, 'IpAddress_to_Country.csv')
    fraud = pd.read_csv(fraud_file, parse_dates=['signup_time', 'purchase_time'])
    ip_df = pd.read_csv(ip_file)
    return fraud, ip_df


def create_ip_dict(df):
    ip_dict = dict()
    for ix, row in df.iterrows():
        ip = row['lower_bound_ip_address']
        ip_dict[ip] = [row['upper_bound_ip_address'], row['country']]
    return ip_dict


def found_lower_bound(arr, tar, ip_dict):
    '''
    found lower bound of an ip by bisection search
    '''
    if tar > 3758096383 or tar < 16777216:
        return None
    n = len(arr)
    if n == 0:
        return None
    if arr[n/2] == tar:
        return tar
    elif arr[n/2] < tar:
        upper = ip_dict[arr[n/2]][0]
        if upper >= tar:
            return arr[n/2]
        elif upper < tar:
            return found_lower_bound(arr[n/2 + 1:], tar, ip_dict)
    elif arr[n/2] > tar:
        return found_lower_bound(arr[:n/2], tar, ip_dict)


def get_country(x, ip_df, ip_dict):
    '''
    convert a ip address to country name
    return county name
    '''
    lower_bound = found_lower_bound(ip_df['lower_bound_ip_address'].values, x, ip_dict)
    if lower_bound in ip_dict:
        return ip_dict[lower_bound][1]


def device_feature(df):
    gp = df.groupby(['device_id'])
    mul_ip = (gp.apply(lambda x:x['ip_address'].unique().shape[0]) > 1).astype(int)
    mul_user = (gp.apply(lambda x:x['user_id'].unique().shape[0]) > 1).astype(int)
    mul_country = (gp.apply(lambda x:x['country'].unique().shape[0]) > 1).astype(int)
    df_new = pd.DataFrame({'multiple_ip': mul_ip,
                           'multiple_user_dev': mul_user,
                           'multiple_country': mul_country}).reset_index()
    return df_new


def ip_feature(df):
    gp = df.groupby(['ip_address'])
    mul_user = (gp.apply(lambda x: x['user_id'].unique().shape[0]) > 1).astype(int)
    df_new = pd.DataFrame({'multiple_user_ip': mul_user}).reset_index()
    return df_new


def column_encode(df, cols):
    encoder_dict = {}
    for col in cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoder_dict[col] = encoder.classes_
    return df, encoder_dict


def data_prepare(data_path):
    # load data
    print 'loading data'
    fraud, ip_df = load_data(data_path)

    # convert ip address into
    print 'getting country'
    ip_dict = create_ip_dict(ip_df)
    fraud['country'] = fraud['ip_address'].apply(lambda x: get_country(x, ip_df, ip_dict))

    # feature engineering
    print 'adding features'
    device = device_feature(fraud)
    ip_fea = ip_feature(fraud)
    fraud = fraud.merge(device, how='left', on='device_id')
    fraud = fraud.merge(ip_fea, how='left', on='ip_address')
    fraud['time_diff_sec'] = (fraud['purchase_time'] - fraud['signup_time']).dt.total_seconds()
    fraud['signup_mon'] = fraud['signup_time'].dt.month
    fraud['purchase_mon'] = fraud['signup_time'].dt.month
    fraud, encoder_dict = column_encode(fraud, ['source', 'browser', 'sex', 'country'])

    # transfer to training data
    droplist = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class']
    X = fraud.drop(droplist, axis=1).values
    y = fraud['class'].values
    features = fraud.drop(droplist, axis=1).columns
    return X, y, features, encoder_dict


def get_metrics(mod, X_test, y_test):
    y_pred = mod.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    ruc = roc_auc_score(y_test, y_pred)
    print 'f1: {:.4f}\tprecision:{:.4f}\trecall: {:.4f}'.format(f1, prec, rec)
    print 'accuracy: {:.4f}\tauc: {:.4f}'.format(acc, ruc)


def plot_roc(y_test, y_pred, y_prob):
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    fig, ax1 = plt.subplots(figsize=(10, 10))
    l1, = ax1.plot(fpr, tpr, color='C1', alpha=0.8,
             lw=2, label='ROC curve (area = %0.2f)' % auc)
    ax1.plot([0, 1], [0, 1], color='C0', linestyle='--')
    ax2 = ax1.twinx()
    l2, = ax2.plot(fpr, thresholds, color='C2', label='threshold', alpha=0.8)

    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax2.set_ylabel('Threshold')

    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    ax1.yaxis.label.set_color(l1.get_color())
    ax2.yaxis.label.set_color(l2.get_color())

    ax1.tick_params('y', colors=l1.get_color())
    ax2.tick_params('y', colors=l2.get_color())

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="right")

    plt.title('Roc Curve and Threshold tuning')


def plot_prec_rec(y_test, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()
    l1, = ax1.plot(recall, precision, lw=2, color='C1',
                   label='Precision-Recall curve')
    l2, = ax2.plot(recall[:-1], thresholds, color='C2', label='Threshold')
    ax1.plot(recall, 1-recall, color='C0', linestyle='--')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax2.set_ylabel('Threshold')

    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left")

    plt.title('Precision-Recall')


def plot_fea_imp(fea_imp, features):
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(features)), np.sort(fea_imp), alpha=0.8)
    plt.yticks(np.arange(len(features)), features[np.argsort(fea_imp)])
    plt.title('Feature Importance')


def model_build(clf, X_train, y_train, X_test, y_test, features, undersample=False, **kwarg):
    if undersample:
        X_train, y_train = RandomUnderSampler().fit_sample(X_train, y_train)
    model = clf(**kwarg)
    model.fit(X_train, y_train)
    get_metrics(model, X_test, y_test)
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    plot_roc(y_test, y_pred, y_prob[:, 1])
    plot_prec_rec(y_test, y_prob[:,1])
    fea_imp = model.feature_importances_
    plot_fea_imp(fea_imp, features)
    return model


if __name__ == '__main__':
    data_path = os.curdir
    X, y, features, encoder_dict = data_prepare(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = model_build(XGBClassifier, X_train, y_train, X_test, y_test, features, undersample=True)
    plt.show()
