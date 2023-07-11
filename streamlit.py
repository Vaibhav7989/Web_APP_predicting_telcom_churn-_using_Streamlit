import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format
import matplotlib.pyplot as plt
# Create Train & Test Data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pickle
#st.write("hello")

! pip install streamlit -q

%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}


def main():
	st.title("Telcom Churn Prediction")
	st.subheader("Enter the details and predict if the customer will churn or not")
	gender=("Male","Female","Prefer not to say")
	gender = st.selectbox("Gender", gender)
	partner = st.radio("Do you have a partner like husband/wife?",('Yes','No'))
	phone = st.radio("Do you have phone service plan?",('Yes','No'))
	Tenure = st.slider("What is the tenure like in years?", 0, 50, 1)
	charge = st.number_input('What are the total charges you are paying for the company?',step=0.01)
	st.write('The current number is ', charge)
	security= st.radio("Do you have a online security like number protection, etc?",('Yes','No'))
	contract=st.radio("What type of contract you are in?",('Contract_Month-to-month','Contract_One year','Contract_Two year'))
	tech_support=st.radio("Are you satisfied or Is tech support provided?",('Yes','No'))
	payment_meth=st.radio("What type of payment method you bhave used?",('PaymentMethod_Bank transfer','PaymentMethod_Credit card','PaymentMethod_Electronic check','PaymentMethod_Mailed check'))
	st.write("the selected are",gender,partner,phone,Tenure,charge,security,contract,tech_support,payment_meth)
	gender_Male=0
	gender_Female=0
	partner_Yes=0
	partner_No=0
	phone_Yes=0
	phone_No=0
	tech_Yes=0
	tech_No=0
	pay_bank=0
	pay_credit=0
	pay_elec=0
	pay_mail=0
	contract_month=0
	contract_one_year=0
	contract_two_year=0
	security_Yes=0
	security_No=0
	tech_No_internet_service=0
	sec_No_internet_service=0

	ok = st.button("Predict Chur")

	if ok:
		#st.write("churn")
		if security=='Yes':
		  sec_Yes=1
		if security=='No':
		  sec_No=1
		if gender=='Female':
		  gender_Female=0
		if gender=='Male':
		  gender_Male=1
		if partner=='Yes':
		  partner_Yes=1
		if partner=='No':
		  partner_No=1
		if tech_support=='Yes':
		  tech_Yes=1
		if tech_support=='No':
		  tech_No=1
		if contract=='Contract_One year':
		  contract_one_year=1
		if contract=='Contract_Two year':
		  contract_two_year=1
		if contract=='Contract_Month-to-month':
		  contract_month=1
		if payment_meth=='PaymentMethod_Bank transfer':
		  pay_bank=1
		if payment_meth=='PaymentMethod_Credit card':
		  pay_credit=1
		if payment_meth=='PaymentMethod_Electronic check':
		  pay_elec=1
		if payment_meth=='PaymentMethod_Mailed check':
		  pay_mail=1
		y=[[Tenure,gender_Female,gender_Male,partner_No,partner_Yes,phone_No,phone_Yes,tech_No,tech_No_internet_service,tech_Yes,contract_month,contract_one_year,contract_two_year,pay_bank,pay_credit,pay_elec,pay_mail,security_No,sec_No_internet_service,security_No,charge]]
		df = pd.DataFrame(y, columns=['Tenure','gender_Female','gender_Male','partner_No','partner_Yes','phone_No','phone_Yes','tech_No','tech_No internet service','tech_Yes','contract_Month-to-month','contract_One year','contract_Two year','pay_Bank transfer (automatic)','pay_Credit card (automatic)','pay_Electronic check','pay_Mailed check','sec_No','sec_No internet service','sec_Yes','charge'])
		df2 = pd.read_csv("churn.csv")
		st.write("The values you have selected are:")
		st.write(df.head())
		df=df.drop(['charge'],axis=1)
		df2.dropna(inplace = True)
		df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
		df2['Churn'].replace(to_replace='No',  value=0, inplace=True)
		df3=df2['charge']
		df2=df2.drop(['charge'],axis=1)
		df2=pd.get_dummies(df2)
		#df2['charge']=df3
		#st.write(df2.head())
		y = df2['Churn'].values
		X = df2.drop(columns = ['Churn'])
		#st.write(X)
		#X['charge']=df3
		model = SGDClassifier()
		model.fit(X, y)
		preds = model.predict(df)
		if preds==0:
		  st.write("The predicted is Not Churn")
		if preds==1:
		  st.write("The predicted is Churn")
		st.write(preds)
if __name__ == '__main__':
	main()

! pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("2NoQxNH5rYDFIyhaLwB5bOqkHyN_2ysdvG3ToLFuTtWzk7bRr") #ngrok.com
!nohup streamlit run app.py --server.port 80 &
url = ngrok.connect(port = '80')
print(url)
