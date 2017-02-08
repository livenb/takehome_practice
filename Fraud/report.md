# Identify Fraud
### James Chen

### Data Sanity check
Data for this practice are coming from two table, the fraud data and the ip to country information. Both data are all well structured no miss value appeared, and the numbers are lying in reasonable range.  

### IP address to country name
There are 138846 different ip range in the ip-country table, if we use naive way to check compare target ip with lower and upper bound, the computation will be expensive, with O(M*N). By construct a hash table and apply a bisection search, the time complexity can be redueced to O(M * logN). Details of the implemention can be found in `src.py` with the function of `create_ip_dict()`, `found_lower_bound()` and `get_country()`.


### Feature engineer
1. By check the data, i found out a strong indication that if a user make purchase right after sign up account, this user will be a fraud. There are 7600 of user_id with time difference of 1 second, *100%* of them are fraud.  
2. IP address and device ID along will be less useful in prediction model, so i tried dig into the relevant features. Consider one device be used on several users or several IPs, this device could be used as fraud. Similarly, count if one ip_address have been used for multiple users or multiple devices.
<center>
<img src="mul_user_dev.png" alt="dev" style="width: 200px;"/>
<img src="mul_user_ip.png" alt="ip" style="width: 200px;"/>
</center>

3. All categorical variable have be transfered to continues integer.


### Model selection
Since large number of categorical features involved, I decide to use the non-linear tree based models, like random forest and extreme gradient booster, which can natively adapt categorical variable. The data are imbalanced, 90% of data showed negative to fraud. A random under sampling technical have been applied on training set.  
Since the model can not perfectly predict the fraud, the model should be choose based on the the cost of false positive and false negative. The metrics of precision is the true positive versus all positive prediction, in this case, precision means how many of the fraud predicted by model are true fraud. Recall is the true positive versus all true fraud, in this case means how many of the fraud have been caught by model.
1. If the false positive cost are low, or the false negative cost are high, we want catch as many fraud as possible, hence choose model based on high recall.
2. If the false positive cost are high, or the false negative cost are low, we want predict the fraud as precision as possible, then model has to be conservative, only predict fraud when possibility is high. At this time, model should be chosen by higher  precision.  
3. There are two way to tuning the precision-recall trade off. One is tuning model parameters targeting higher precision/recall, another way is tuning threshold by roc curve or precision recall curve.  
<img src="roc_xgb_undersample.png" alt="roc" style="width: 230px;"/>
<img src="pre_rec_xgb_undersample.png" alt="roc" style="width: 230px;"/>

As shown in above roc curve and precision recall curve, the threshold are green lines, the desired ratio of precision / recall can be achieved by choose different threshold.  

### User experience
User experience will be affect by different model selection.
1. for a high precision model, less non-fraud users will be affect by fraud detection and meanwhile more fraud will be missed. User will have better exprience.
2. for a high recall model, more fraud alarm will be made, hence more users will be affect. The user experience will be less comfortable.

### Model interpretation
One import result to choose the ensemble tree based model is the feature importance can be given by the ratio of information explained by a specific variable. In our case, the four most important features are time between sign up and purchase, if a ip have been used by multiple users, if a device have been used by multiple users, and the purchase.

<center><img src="fea_imp.png" alt="feature importance" style="width: 350px;"/></center>

A typical fraud can be portrayed as: the purchase is made immediately after sign up an account, the ip address have been used by multiple user id, multiple user ids used same device and usually purchase price between 30 and 40.

### Future work
A more dedicate feature engineer should be a applied and explored the higher order interaction between the variables. The model parameters should be fine tuned by the desire metrics.
