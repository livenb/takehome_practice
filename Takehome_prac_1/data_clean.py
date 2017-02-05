import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def read_data(data_path='./'):
    '''
    read data from csv files, and join all 3 table together
    transfer the
    '''
    email_opened = pd.read_csv(data_path + 'email_opened_table.csv')
    emails = pd.read_csv(data_path + 'email_table.csv')
    clicked = pd.read_csv(data_path + 'link_clicked_table.csv')
    df = emails.merge(email_opened, how='left', on='email_id',
                      indicator='email_opened')
    df = df.merge(clicked, how='left', on='email_id', indicator='link_clicked')
    df['email_opened'] = df['email_opened'].apply(modify_indictator)
    df['email_opened'] = df['email_opened'].astype(int)
    df['link_clicked'] = df['link_clicked'].apply(modify_indictator)
    df['link_clicked'] = df['link_clicked'].astype(int)
    return df


def modify_indictator(row):
    if row == 'left_only':
        return 0
    elif row == 'both':
        return 1
    else:
        return -1


def clean_data(df):
    '''
    There exists 50 link-clicked without email-open
    which has to be eliminate from the data
    '''
    mask = np.logical_not(df.isin(df[(df['email_opened'] == 0)
                                     & (df['link_clicked'] == 1)]))
    df_clean = df[mask].dropna().copy()
    return df_clean


def combine_email_cat(row):
    '''
    make combination of email text length and email version
    '''
    return row['email_version'] + '_' + row['email_text']


def transfer_weekday(row):
    '''
    add num to weekday, make it easier to sort in sequence
    '''
    week_dict = {'Sunday': '7 Sunday', 'Wednesday': '3 Wednesday',
                 'Monday': '1 Monday', 'Saturday': '6 Saturday',
                 'Friday': '5 Friday', 'Tuesday': '2 Tuesday',
                 'Thursday': '4 Thursday'}
    return week_dict[row]


def segment_purchase(row):
    '''
    seperate the user purchased record into 5 categories:
    0: no purchase
    1: 1-5
    2: 6-10
    3: 11-15
    4: more than 15
    '''
    if row == 0:
        return '0: no purchase'
    elif row >= 1 and row <= 5:
        return '1: 1-5'
    elif row >= 6 and row <= 10:
        return '2: 6-10'
    elif row >= 11 and row <= 15:
        return '3: 11-15'
    else:
        return '4: more than 15'


def plot_cat(col, df, ax):
    gp = df.groupby(col)[['email_opened', 'link_clicked']]
    df1 = (gp.sum() / gp.count()).reset_index()
    df2 = gp.sum().reset_index()
    df1['open_to_click'] = df2['link_clicked'] / df2['email_opened']
    df1.index = df1[col]
    df1.drop(col, axis=1).transpose().plot.bar(ax=ax, rot=0)


def plot_num(col, df, ax):
    num = len(df[col].unique())
    df[col][(df['link_clicked'] == 1)].hist(bins=num, normed=1, alpha=0.5,
                                            label='link clicked', ax=ax)
    df[col][(df['link_clicked'] == 0)].hist(bins=num, normed=1, alpha=0.5,
                                            label='not clicked', ax=ax)
    ax.legend()
    ax.set_xlabel(col)


def plot_ctr(col, df, ax):
    gp = df.groupby(col)[['email_opened', 'link_clicked']]
    df2 = gp.sum().reset_index()
    (df2['link_clicked'] / df2['email_opened']).plot.bar(ax=ax, rot=0)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(1, 25))
    ax.set_xlabel(col)
    ax.legend(['Open to click rate'], loc='best')


def email_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_cat('email_cat', df, axes[0][0])
    plot_cat('weekday', df, axes[0][1])
    plot_num('hour', df, axes[1][0])
    plot_ctr('hour', df, axes[1][1])
    plt.tight_layout()


def user_eda(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_cat('user_country', df, axes[0])
    plot_num('user_past_purchases', df, axes[1])
    plot_cat('purchase_cat', df, axes[2])
    plt.tight_layout()


def out_put_prepare(df):
    '''
    transfer all category to integer, prepare for machine learning
    '''
    df['weekday'] = df['weekday'].apply(lambda x: int(x.split()[0]))
    df['email_text'] = df['email_text'].apply(lambda x:
                                              1 if x == 'short_email' else 0)
    df['email_version'] = df['email_version'].apply(lambda x:
                                                    1 if x == 'personalized'
                                                    else 0)
    df['purchase_cat'] = df['purchase_cat'].apply(lambda x:
                                                  int(x.split(':')[0]))
    df = pd.concat([df, pd.get_dummies(df['user_country'])], axis=1)
    country_dict = {'US': 0, 'UK': 1, 'ES': 2, 'FR': 3}
    df['user_country'] = df['user_country'].apply(lambda x: country_dict[x])
    email_dict = {'generic_long_email': 0, 'generic_short_email': 1,
                  'personalized_long_email': 2, 'personalized_short_email': 3}
    df['email_cat'].unique()
    df['email_cat'] = df['email_cat'].apply(lambda x: email_dict[x])
    return df


def data_prepare(data_path='./'):
    df = read_data(data_path)
    df = clean_data(df)
    df['email_cat'] = df.apply(combine_email_cat, axis=1)
    df['weekday'] = df['weekday'].apply(transfer_weekday)
    df['purchase_cat'] = df['user_past_purchases'].apply(segment_purchase)
    email_eda(df)
    user_eda(df)
    df = out_put_prepare(df)
    return df

if __name__ == '__main__':
    data_prepare()
    plt.show()
