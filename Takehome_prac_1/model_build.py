from data_clean import data_prepare
import scipy.stats as sps
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def choose_user(df):
    '''
    Without running a ML model, choose the user group of higher possibility
    '''
    choose, opened, clicked = 0, 0, 0
    for row in df.iterrows():
        row = row[1]
        if (row['email_text'] == 1 and
            row['email_version'] == 1 and
            8 <= row['hour'] <= 14 and
            row['weekday'] <= 4 and
            row['user_past_purchases'] > 0 and
            row['user_country'] <= 1):
            choose += 1
            if row['email_opened'] == 1:
                opened += 1
            if row['link_clicked'] == 1:
                clicked += 1
    print '\n'+'*'*30
    print 'The result of choose user by eda withou ML:'
    print '{}% of email opened'.format(round(opened*1.0 / choose * 100, 2))
    print '{}% of link clicked'.format(round(clicked*1.0 / choose * 100, 2))


def hour_prob():
    '''
    make probalbility list for hour, with a bimodal normal distribution
    '''
    a = sps.norm(12, 4)
    b = sps.norm(23, 3)
    x = range(1, 30)
    p = 3 * a.pdf(x) + b.pdf(x)
    p = list(p[23:28]) + list(p[4: 23])
    p = np.array(p) / sum(p)
    return p


def generate_sim_data(size=20000):
    '''
    Make simulation data by the strategy discribed
    '''
    email_text = np.random.binomial(1, 1, size)
    email_ver = np.random.binomial(1, 1, size)
    p_hour = hour_prob()
    p_purchase = sps.norm(12, 5).pdf(range(0, 23))
    p_purchase = p_purchase / p_purchase.sum()
    p_weekday = sps.norm(3, 4).pdf(range(1, 8))
    p_weekday = p_weekday/p_weekday.sum()
    hour = np.random.choice(range(1, 25), size, p=p_hour)
    purchase = np.random.choice(range(0, 23), size, p=p_purchase)
    weekday = np.random.choice(range(1, 8), size, p=p_weekday)
    country = np.random.choice(range(4), size, p=[0.7, 0.2, 0.05, 0.05])
    data = np.stack([email_text, email_ver, weekday, purchase, hour, country])
    return data.T


def cv_model(X, y, clf, smp, **kwclf):
    kf = KFold(n_splits=5)
    acc, pre, rec, f1 = 0, 0, 0, 0
    for train_idx, cross_idx in kf.split(X):
        X_t, y_t = X[train_idx], y[train_idx]
        X_c, y_c = X[cross_idx], y[cross_idx]
        X_sampled, y_sampled = smp().fit_sample(X_t, y_t)
        mod = clf(**kwclf).fit(X_sampled, y_sampled)
        y_p = mod.predict(X_c)
        acc += accuracy_score(y_c, y_p)
        pre += precision_score(y_c, y_p)
        rec += recall_score(y_c, y_p)
        f1 += f1_score(y_c, y_p)
    print 'CV accuracy:{}\t precision:{}\t'\
          'recall:{}\tf1:{}'.format(round(acc/5, 4), round(pre/5, 4),
                                    round(rec/5, 4), round(f1/5, 4))


def prediction_model(df, clf, smp, **kwclf):
    '''
    Build prediction model for email open and link click
    '''
    cols_cat = ['email_text', 'email_version', 'weekday',
                'user_past_purchases', 'hour', 'user_country']
    X_1 = df[cols_cat].values
    y_1 = df['email_opened'].values
    print '-'*30
    print 'stage 1'
    cv_model(X_1, y_1, clf, smp, **kwclf)
    X_1, y_1 = smp().fit_sample(X_1, y_1)
    clf_1 = clf(**kwclf)
    clf_1.fit(X_1, y_1)
    X_2 = df[df['email_opened'] == 1][cols_cat].values
    y_2 = df[df['email_opened'] == 1]['link_clicked'].values
    print 'stage 2'
    cv_model(X_2, y_2, clf, smp, **kwclf)
    X_2, y_2 = smp().fit_sample(X_2, y_2)
    clf_2 = clf(**kwclf)
    clf_2.fit(X_2, y_2)
    return clf_1, clf_2


def run_simulation(df, n_times, n_size,  clf, smp, **kwclf):
    clf_1, clf_2 = prediction_model(df, clf, smp, **kwclf)
    score_1, score_2 = 0, 0
    for i in range(n_times):
        X_sim = generate_sim_data(n_size)
        y_pred_1 = clf_1.predict(X_sim)
        y_pred_2 = clf_2.predict(X_sim[y_pred_1 == 1])
        score_1 += y_pred_1.sum() / len(y_pred_1)
        score_2 += y_pred_2.sum() / len(y_pred_2)
    score_1 /= n_times
    score_2 /= n_times
    print '-'*30
    print 'Simulation result:'
    print 'Average email open rate: {}%'.format(round(score_1*100, 2))
    print 'Average open to click rate: {}%'.format(round(score_2*100, 2))


if __name__ == '__main__':
    df = data_prepare()
    choose_user(df)
    print '\n' + '*'*30
    print 'Naive Bayes'
    run_simulation(df, 5, 100000, GaussianNB, RandomUnderSampler)
    print '\n' + '*'*30
    print 'Random Forest'
    params = {'n_estimators': 200}
    run_simulation(df, 5, 100000, RandomForestClassifier,
                   RandomUnderSampler, **params)
    print '\n' + '*'*30
    print 'Extreme Gradient Boost'
    run_simulation(df, 5, 100000, XGBClassifier, RandomUnderSampler, **params)
