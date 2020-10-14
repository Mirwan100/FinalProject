from flask import Flask
from flask import render_template,request
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Create an app object
app = Flask(__name__)

#Whenever this is called it will activate the function
@app.route('/')
def home(): # initialize a web app with flask
	return render_template('prediction.html')
# Render About page
@app.route('/about')
def about():
    return render_template('about.html')
# Render About page
@app.route('/visualization')
def visualization():
    return render_template('visualization.html')
@app.route('/histogram')
def histogram():
    return render_template('histogram.html')

@app.route('/boxplot')
def boxplot():
    return render_template('boxplot.html')

@app.route('/average')
def average():
    return render_template('average.html')
data_le = pd.read_csv(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\data_le.csv')
RFR = joblib.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\finalized_model.sav')

def trainingData(df,n):
    X = df.iloc[:,n]
    y = df.iloc[:,-1:].values.T
    y=y[0]
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,test_size=0.1,random_state=0)
    return (X_train,X_test,y_train,y_test)

X_train,X_test,y_train,y_test=trainingData(data_le,list(range(len(list(data_le.columns))-1)))

def formatrupiah(uang):
    y = str(uang)
    if len(y) <= 3 :
        return 'Rp ' + y     
    else :
        p = y[-3:]
        q = y[:-3]
        return   formatrupiah(q) + '.' + p

#FUNCTION that encode user's input using LabelEncoder and scale using MixMax Scaler
def prep_features(result):
	le = preprocessing.LabelEncoder()
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\band.npy', allow_pickle=True)
	result['band'] = le.transform(result['band'])
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\model.npy', allow_pickle=True)
	result['model'] = le.transform(result['model'])
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\fuelType.npy', allow_pickle=True)
	result['fuelType'] = le.transform(result['fuelType'])
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\bodytype.npy', allow_pickle=True)
	result['bodytype'] = le.transform(result['bodytype'])
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\transmission.npy', allow_pickle=True)
	result['transmission'] = le.transform(result['transmission'])
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\color.npy', allow_pickle=True)
	result['color'] = le.transform(result['color'])
	le.classes_ = np.load(r'C:\Users\tab\Desktop\Modul 3\tugas akhir\place.npy', allow_pickle=True)
	result['place'] = le.transform(result['place'])

	norm = MinMaxScaler()
	result.milage = norm.fit_transform(np.array(result.milage).reshape(-1,1))
	result.band = norm.fit_transform(np.array(result.band).reshape(-1, 1))
	result.model = norm.fit_transform(np.array(result.model).reshape(-1, 1))
	result.place = norm.fit_transform(np.array(result.place).reshape(-1, 1))


	return result


@app.route('/predictprice', methods=['POST','GET']) # this is the actual post request which will take the inputs run the model and hopefully return a prediction
def predict_price():
	print(request.__dict__) # test to check what it is being inputted into request
	if request.method=='POST':
	 	#Storage method for data (requests the web server to accept data)
		cars=request.form
		print(request.form.__dict__)# this form will take the inputs
		band=cars['band']
		model=cars['model']
		year=cars['year']
		fuelType=cars['fuelType']
		bodytype=cars['bodytype']
		transmission=cars['transmission']
		seating=cars['seating']
		milage=cars['milage']
		color=cars['color']
		place=cars['place']
		# the function from above will convert this result dict into a vector of 292 values
		result = {'band':band, 'model':model, 'year':year,'fuelType':fuelType, 'bodytype':bodytype,'transmission':transmission,'seating':seating,'milage':milage,'color':color,'place':place}
		result = pd.DataFrame(result, index=[0])
		print("hey")
		mobil = str(result.band.iloc[0]) + " " + str(result.model.iloc[0]) + " "+ str(result.year.iloc[0])
		test = prep_features(result)
		print(test)
		prediction = RFR.predict(test)
		# calls result.html which displays the output from the random forest
		prediction = int(np.exp(prediction)[0])
		prediction = str(prediction)
		c = int(prediction[-5:])
		prediction = int(prediction)-c
		prediction = formatrupiah(prediction)
	return render_template('result.html',prediction=prediction, mobil=mobil) # returns prediction
	


app.run(debug=True)
