import pandas as pd
import os

def create_ip_dict(df):
    ip_dict = dict()
    for ix, row in df.iterrows():
        ip = row['lower_bound_ip_address']
        ip_dict[ip] = [row['upper_bound_ip_address'], row['country']]
    return ip_dict

def found_lower_bound(arr, tar):
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
    elif  arr[n/2] < tar:
        upper = ip_dict[arr[n/2]][0]
        if upper >= tar:
            return arr[n/2]
        elif upper < tar:
            return found_lower_bound(arr[n/2 + 1:], tar)
    elif arr[n/2] > tar:
        return found_lower_bound(arr[:n/2], tar)

def get_country(x):
    lower_bound = found_lower_bound(ip['lower_bound_ip_address'].values, x)
    if lower_bound in ip_dict:
        return ip_dict[lower_bound][1]

def device_feature():
    gp = fraud.groupby(['device_id'])
    mul_ip = (gp.apply(lambda x:x['ip_address'].unique().shape[0]) > 1).astype(int)
    mul_user = (gp.apply(lambda x:x['user_id'].unique().shape[0]) > 1).astype(int)
    mul_country = (gp.apply(lambda x:x['coutry_test'].unique().shape[0]) > 1).astype(int)
    df = pd.DataFrame({'mutiple_ip': mul_ip,
                       'multiple_user_dev': mul_user,
                       'multiple_country': mul_country}).reset_index()
    return df

def ip_feature():
    gp = fraud.groupby(['ip_address'])
    mul_user = (gp.apply(lambda x:x['user_id'].unique().shape[0]) > 1).astype(int)
    df = pd.DataFrame({'multiple_user_ip': mul_user}).reset_index()
    return df

def data_prepare():
    fraud_file = os.path.join(os.curdir, 'Fraud_Data.csv')
    ip_file = os.path.join(os.curdir, 'IpAddress_to_Country.csv')
