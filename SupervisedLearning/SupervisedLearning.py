#!/usr/bin/env python
# coding: utf-8

# Data Description:
# The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.
# 
# Domain:Banking
# 
# Context:
# This case is about a bank (Thera Bank) whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.
# 
# Learning Outcomes:
# 
# Exploratory Data Analysis
# Preparing the data to train a model
# Training and making predictions using a classification model
# Model evaluation
# Objective:
# The classification goal is to predict the likelihood of a liability customer buying personal loans.
# 
# Steps and tasks:
# 
# Read the column description and ensure you understand each attribute well
# Study the data distribution in each attribute, share your findings
# Get the target column distribution.
# Split the data into training and test set in the ratio of 70:30 respectively
# Use different classification models (Logistic, K-NN and Naïve Bayes) to predict the likelihood of a liability customer buying personal loans
# Print the confusion matrix for all the above models
# Give your reasoning on which is the best model in this case and why it performs better?
# References:
# 
# Data analytics use cases in Banking
# Machine Learning for Financial Marketing
# 

# In[52]:


import pandas as pd
import matplotlib.pyplot as plt


# In[53]:


df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df


# In[54]:


df.corr()


# In[55]:


print(df.dtypes)


# In[56]:


print(df.isnull().sum())


# In[57]:


print(df.isnull().any(axis=0))


# In[58]:


correlation_matrix = df.corr()
threshold = 0.3

print("significant correlations: ")
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        # print(abs(correlation_matrix.iloc[i,j]))
        if abs(correlation_matrix.iloc[i,j]) > threshold:
            colname1 = correlation_matrix.columns[i]
            colname2 = correlation_matrix.columns[j]
            correlation_value = correlation_matrix.iloc[i,j]
            correlation_type = "positive" if correlation_value > 0 else "negative"
            print(f" columns '{colname1}' and {colname2}' have a {correlation_type} of {correlation_value:.2f}")


# #### Recursive Feature Elemination
# 

# In[59]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[60]:


#Selecting features based on Correlation Threshold
threshold = 0.1
# Get the list of columns
columns = correlation_matrix.columns

# Create a set to keep track of features with significant correlations
selected_features = set()

# Iterate through the matrix to find correlations exceeding the threshold
for i in range(len(columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname1 = columns[i]
            colname2 = columns[j]
            correlation_value = correlation_matrix.iloc[i, j]
            print(f"Columns '{colname1}' and '{colname2}' have a correlation of {correlation_value:.2f}")
            selected_features.add(colname1)
            selected_features.add(colname2)

# Convert the set to a list for further use
selected_features = list(selected_features)
print("Selected Features based on Correlation Threshold:")
print(selected_features)


# In[61]:


y = df['Personal Loan']
y


# In[67]:


# Define a list of columns to remove
columns_to_remove = ['Personal Loan']

# Remove the specified columns from the list of selected features
selected_features = [feature for feature in selected_features if feature not in columns_to_remove]


# In[68]:


X = df[selected_features]


# In[69]:


X


# In[73]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# #### Split data
# 

# In[78]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# #### Scale data
# 

# In[79]:


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #### Select model and fit
# 

# In[80]:


# Initialize and fit the Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs')  # Increase max_iter as needed
model.fit(X_train_scaled, y_train)


# #### evaluate model
# 

# In[81]:


# Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# ### Confusion Matrix
# 

# In[82]:


from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# In[83]:


# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[84]:


# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# ## Prediction Application
# 
# The prediction application is a simple application that uses the trained model to make predictions on new, unseen
# 

# In[85]:


import joblib


# In[86]:


# Save the model and scaler
joblib.dump(model, 'logistic_regression_model.joblib')
joblib.dump(scaler, 'scaler.joblib')


# In[87]:


def input_customer_details():
    print("Enter new customer details:")
    age = int(input("Age: "))
    experience = int(input("Experience: "))
    income = float(input("Income: "))
    ccavg = float(input("CCAvg: "))
    mortgage = float(input("Mortgage: "))
    return pd.DataFrame([[age, experience, income, ccavg, mortgage]], columns=selected_features)


# In[88]:


def load_model_and_predict(new_data):
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler.joblib')
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)
    return prediction, probability


# In[90]:


import matplotlib.pyplot as plt

def suggest_improvements(new_data):
    # Example suggestions (you can tailor these based on your domain knowledge)
    improvements = []
    if new_data['Income'].values[0] < 60000:
        improvements.append("Increase your income.")
    if new_data['CCAvg'].values[0] > 2:
        improvements.append("Reduce your average credit card spending.")
    if new_data['Mortgage'].values[0] > 2000:
        improvements.append("Reduce your mortgage.")
    return improvements

def display_graphical_representation(prediction, probability):
    plt.figure(figsize=(6, 4))
    plt.bar(['No Loan', 'Loan'], probability[0], color=['blue', 'orange'])
    plt.title('Loan Prediction Probability')
    plt.ylabel('Probability')
    plt.show()

def main():
    new_data = input_customer_details()
    prediction, probability = load_model_and_predict(new_data)
    
    result = 'will get' if prediction[0] == 1 else 'will not get'
    print(f"The customer {result} a personal loan.")
    display_graphical_representation(prediction, probability)
    
    improvements = suggest_improvements(new_data)
    if improvements:
        print("Suggestions for improvement:")
        for suggestion in improvements:
            print(f"- {suggestion}")
    else:
        print("No specific improvements needed.")

if __name__ == "__main__":
    main()


# In[ ]:




