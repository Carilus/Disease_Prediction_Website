"""
Routes and views for the flask application.
"""

#from crypt import methods
#from crypt import methods
from datetime import datetime
from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from flask import render_template, request, url_for, redirect,session,flash
from Disease_Prediction_Website import app
from flask_mail import Message,Mail
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import time

mymail = Mail(app)

engine  = create_engine('mysql+mysqldb://root:@localhost/disease_prediction?charset=utf8mb4', echo=None)
    #Creating a session
#session = sessionmaker(bind=engine)()
Base = declarative_base()

@app.route('/')
def entry():
    """Renders the history page."""
    return render_template(
        'home.html',
        title='',
        year=datetime.now().year,
        message='Welcome.'
    )

@app.route('/home')
def home():
    """Renders the home page."""
    if session.get('email') != None:
        return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )
    else:
        flash("Login required")
        return redirect(url_for('login'))

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact Us',
        year=datetime.now().year,
        message='Mail us.'
    )
@app.route('/services')
def services():
    """Renders the contact page."""
    return render_template(
        'services.html',
        title='Services',
        year=datetime.now().year,
        message='Our Services.'
    )
@app.route('/system_services')
def system_services():
    """Renders the contact page."""
    return render_template(
        'system_services.html',
        title='Services',
        year=datetime.now().year,
        message='Our Services.'
    )
@app.route('/contact_us')
def contact_us():
    """Renders the contact page."""
    return render_template(
        'contact_us.html',
        title='Contact Us',
        year=datetime.now().year,
        message='Mail us.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Welcome.'
    )
@app.route('/about_system')
def about_system():
    """Renders the about page."""
    return render_template(
        'about_system.html',
        title='About',
        year=datetime.now().year,
        message='Welcome.'
    )

@app.route('/predict')
def predict():
    """Renders the about page."""
    if session.get('email') != None:
        return render_template(
            'predict.html',
            title='About',
            year=datetime.now().year,
            message='Your application description page.'
        )
    else:
        flash("Login required")
        return redirect(url_for('login'))

samples =[]
@app.route('/predictions',methods=['POST','GET'])
def predictions():
    if session.get('email') != None:
        if request.method == "POST":
            #Importing data
            data = pd.read_csv('dataset.csv')

            symptoms = pd.read_csv('Symptom-severity.csv',header=0)
            precautions = pd.read_csv('symptom_precaution.csv')
            description = pd.read_csv('symptom_Description.csv')
            cols = data.columns
            df = data[cols].values.flatten()
            s = pd.Series(df)
            s = s.str.strip()
            s = s.values.reshape(data.shape)
            
            #creating a new clean dataset
            data = pd.DataFrame(s, columns=data.columns)
            data = data.fillna(0)
            
            #Extracting values of the dataframe
            vals = data.values
            sympts = symptoms['Symptom'].unique()
            symptsID = symptoms['ID']
            
            for i in range(len(sympts)):
                vals[vals==sympts[i]] = symptoms[symptoms['Symptom']==sympts[i]]['weight'].values[0]
            
            
            data1 = pd.DataFrame(vals,columns=cols)
            
            #Giving symptoms not included in sumptoms datasets IDs
            data1 = data1.replace('dischromic _patches',0)
            data1 = data1.replace('foul_smell_of urine',0)
            data1 = data1.replace('spotting_ urination',0)
            data1 = data1.replace('foul_smell_of urine',0)
            
            #Labels and Features
            features = data1.iloc[:,1:]
            labels =data1['Disease'].values
            
            #Test Train Split
            x_train,x_test,y_train,y_test=train_test_split(features,labels,train_size=0.85,shuffle =True)
            
            #Model SVC
            Model_SVC = SVC()
            Model_SVC.fit(x_train, y_train)
            
            #Test
            predicted =Model_SVC.predict(x_test)
            
            accuracy_score(y_test,predicted)
            
            #Model RFC
            RFC = RandomForestClassifier()
            RFC.fit(x_train,y_train)
            
            predt = RFC.predict(x_test)
            
            accuracy = accuracy_score(y_test,predt) #1.0
            #mpredict
            dat = np.array((10,22,3,4,5,0,30,11,0,0,0,40,0,12,34,0,0))
            
            predictip = Model_SVC.predict(dat.reshape(1,-1))
            pdtip  =RFC.predict(dat.reshape(1,-1))
            s1 = request.form.get('s1',type=int)
            s2 = request.form.get('s2',type=int)
            s3 = request.form.get('s3',type=int)
            s4 = request.form.get('s4',type=int)
            s5 = request.form.get('s5',type=int)
            s6 = request.form.get('s6',type=int)
            s7 = request.form.get('s7',type=int)
            s8 = request.form.get('s8',type=int)
            s9 = request.form.get('s9',type=int)
            s10 = request.form.get('s10',type=int)
            s11 = request.form.get('s11',type=int)
            s12 = request.form.get('s12',type=int)
            s13 = request.form.get('s13',type=int)
            s14 = request.form.get('s14',type=int)
            s15 = request.form.get('s15',type=int)
            s16 = request.form.get('s16',type=int)
            s17 = request.form.get('s17',type=int)
            samples2 = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17]
            samples = np.array(samples2)
            samples1 = np.array(samples2)
            
            #Preprocessing input: Relacing symptom Ids with symptom weights
            for i in range(len(symptsID)):
                samples[samples==symptsID[i]] = symptoms[symptoms['ID']==symptsID[i]]['weight'].values[0]
            #Predicting the disease
            disease = Model_SVC.predict(samples.reshape(1,-1))

            #Saving predictions
            disease_predicted = disease[0]
            accuracy_of_prediction = str(accuracy*100)
            add_user_to_prediction_history(disease_predicted,accuracy_of_prediction)
            
            status = all(i==0 for i in samples)
            
            result = disease[0]

            if(7 in samples1 and 12 in samples1 and 26 in samples1 and 32 in samples1 and 5 in samples1 and 36 in samples1):
                result = "Malaria"
            if(4 in samples1 and 25 in samples1 and 28 in samples1 and 57 in samples1 and 89 in samples1 and 108 in samples1 and 109 in samples1 and 55 in samples1 and 51 in samples1 and 57 in samples1):
                result = "Tuberculosis"
            if(6 in samples1 and 15 in samples1 and 26 in samples1 and 28 in samples1 and 25 in samples1 and 46 in samples1 and 57 in samples1 and 59 in samples1 and 109 in samples1 and 51 in samples1):
                result = "Pneumonia"
            if(7 in samples1 and 12 in samples1 and 33 in samples1 and 34 in samples1 and 35 in samples1 and 36 in samples1 and 40 in samples1 and 41 in samples1 and 42 in samples1 and 44 in samples1 and 98 in samples1):
                result = "hepatitis A"
            if(4 in samples1 and 25 in samples1 and 26 in samples1 and 32 in samples1 and 52 in samples1 and 53 in samples1 and 55 in samples1 and 89 in samples1):
                result = "Common Cold"
            if(100 in samples1 and 1 in samples1 and 2 in samples1 and 32 in samples1 and 26 in samples1 and 48 in samples1 and 26 in samples1 and 49 in samples1):
                result = "Chicken pox"
            if(6 in samples1 and 12 in samples1 and 15 in samples1 and 26 in samples1 and 39 in samples1 and 40 in samples1 and 41 in samples1 and 35 in samples1):
                result = "Typhoid"

            desc = description[description['Disease']==result]['Description'].values[0]
            prec1 = precautions[precautions['Disease']==result]['Precaution_1'].values[0]
            prec2 = precautions[precautions['Disease']==result]['Precaution_2'].values[0]
            prec3 = precautions[precautions['Disease']==result]['Precaution_3'].values[0]
            prec4 = precautions[precautions['Disease']==result]['Precaution_4'].values[0]
            session['disease'] = result

            if(status == False):
                return render_template('result.html',result=result,accuracy=accuracy*100,desc=desc,prec1=prec1,prec2=prec2,prec3=prec3,prec4=prec4)
                
            else:
                return render_template('normal.html')
    else:
        flash("Login required")
        return redirect(url_for('login'))
      


@app.route('/get_doctor')
def doctor():
    """Renders the doctors page."""
    if session.get('email') != None:
        return render_template(
            'doctors.html',
            title='Doctors',
            year=datetime.now().year,
            message='Welcome.'
        )
    else:
        flash("Login required")
        return redirect(url_for('login'))

@app.route('/retrieve_doctor', methods = ['POST','GET'])
def retrieve_doctor():
    """Renders the doctors page."""
    if session.get('email') != None:
        country = request.form.get('country',type=str)
        result = engine.execute("SELECT id, name, location, email FROM doctors WHERE country = '"+str(country)+"';")
        return render_template(
            'doctors.html',
            result=result,
            title='Doctors',
            year=datetime.now().year,
            message='Welcome.'
        )
    else:
        flash("Login required")
        return redirect(url_for('login'))


@app.route('/medical_history')
def history():
    """Renders the history page."""
    if session.get('email') != None:
        result = engine.execute("SELECT id, disease, accuracy, date FROM prediction_history WHERE email = '"+session.get('email')+"';")
        return render_template(
            'history.html',
            result = result,
            title='Prediction History',
            year=datetime.now().year,
            message='Welcome.'
        )
    else:
        flash("Login required")
        return redirect(url_for('login'))


@app.route('/model_info')
def model():
    """Renders the history page."""
    return render_template(
        'model.html',
        title='',
        year=datetime.now().year,
        message='Welcome.'
    )
@app.route('/login')
def login():
    """Renders the login page."""
    return render_template(
        'login.html',
        title='',
        year=datetime.now().year,
        message='Login.'
    )
@app.route('/register')
def register():
    """Renders the register page."""
    return render_template(
        'register.html',
        title='',
        year=datetime.now().year,
        message='Register.'
    )

@app.route('/login_validation', methods = ['POST','GET'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    return "The email is {} and password is {}".format(email,password)

@app.route('/registeration_to_db', methods = ['POST','GET'])
def registeration_to_db():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    try:
        add_user_to_database(name,email,password)
    except:
        return redirect(url_for('register', message = "User already exist"))
        
    
    return redirect(url_for('login', email=email))

    

#Add Users to the database
def add_user_to_database(name, email, password):
    engine.execute("INSERT INTO users(name,email,password) VALUES('"+name+"','"+email+"','"+password+"');")
    
#Retrive user
@app.route('/user_login', methods = ['POST','GET'])
def user_login():
    email = request.form.get('email')
    password = request.form.get('password')
    try:
        result = engine.execute("SELECT email, password FROM users WHERE email = '"+email+"';")
        user = [dict(row) for row in result.fetchall()]
        SavedEmail =  user[0]['email']
        SavedPassword = user[0]['password']

        
        if(email == SavedEmail and password == SavedPassword):
            session['loggedin'] = True
            session['email'] = SavedEmail
            session['password'] = password
            session['room'] = 'Doctors'
            return redirect(url_for("home"))
        else:
            return redirect(url_for('login', message = "Invalid Email or Password"))
    except:
        return redirect(url_for("login", message = "Error accessing the server"))

@app.route('/logout')
def logout():
    if session.get('email') != None:
        session.clear()
        return redirect(url_for('login'))
    else:
        flash("Login required")
        return redirect(url_for('login'))

def login_required():
    if session.get('email') == None:
        flash('Login required')
        return redirect(url_for('login'))

#Add Users to the database
def add_user_to_prediction_history(disease, accuracy):
    engine.execute("INSERT INTO prediction_history(disease,accuracy,email) VALUES('"+disease+"','"+accuracy+"','"+session.get('email')+"');")

#Retrieve history
def retrieve_predictions():
    try:
        result = engine.execute("SELECT id, disease, accuracy FROM prediction_history WHERE email = '"+email+"';")
        user = [dict(row) for row in result.fetchall()]

    except:
        pass

@app.route("/mail", methods=["GET", "POST"])
def mail():
    sender = 'cariluslinnaeus@gmail.com'
    recipients = ['oselucarilus1@gmail.com']
    if request.method == "POST":
        subject = request.form.get("subject")
        body = request.form.get("body")
        mymail.send_message(sender=sender, recipients=recipients, subject=subject, body=body)
        flash("Sent Successfully")
    return render_template("send_mail.html")

@app.route('/send_mail')
def send_mail():
    return render_template('send_mail.html')

@app.route('/drugs')
def drugs():
    drugs = pd.read_csv('drugs.csv')
    disease = session.get('disease')
    drug1 = drugs[drugs['Disease']==disease]['Drug_1'].values[0]
    drug2 = drugs[drugs['Disease']==disease]['Drug_2'].values[0]
    drug3 = drugs[drugs['Disease']==disease]['Drug_3'].values[0]
    drug4 = drugs[drugs['Disease']==disease]['Drug_4'].values[0]
    drug6 = drugs[drugs['Disease']==disease]['Drug_6'].values[0]
    drug7 = drugs[drugs['Disease']==disease]['Drug_7'].values[0]
    drug8 = drugs[drugs['Disease']==disease]['Drug_8'].values[0]

    return render_template("drugs.html",disease=disease, drug1=drug1,drug2=drug2,drug3=drug3,drug4=drug4,drug6=drug6,drug7=drug7,drug8=drug8)

@app.route('/password_reset')
def password_reset():
    return render_template('password_reset.html')

app.route('/reset_password')
def reset_password():
    return render_template('login.html')

@app.route("/send_predicted_disease")
def send_predicted_disease():
    sender = session.get('email')
    recipients = ['oselucarilus1@gmail.com']
    if request.method == "POST":
        subject = "Further Treatment"
        body = "Hello Dr, I am "+session.get('name')+ ". I recently used the Disease Prediction System to predict my condition. The outcome was " + session.get('disease')+ ". I would like to get further help from you. Thank you." 
        mymail.send_message(sender=sender, recipients=recipients, subject=subject, body=body)
        flash("Sent Successfully")
        flash("Sent Successfully")
        time.sleep(5)
    return render_template("send_mail.html")

@app.route("/send_prediction_history")
def send_prediction_history():
    sender = session.get('email')
    recipients = ['oselucarilus1@gmail.com']
    result = engine.execute("SELECT id, disease, accuracy, date FROM prediction_history WHERE email = '"+session.get('email')+"';")

    if request.method == "POST":
        subject = "Further Treatment"
        body = "Hello Dr, I am "+session.get('name')+ ". I need help from you. I`ve been using the Disease Prediction System and the following is my prediction history."+"<ul>" + [dict(i)['disease'] for i in result]+"</ul>"
        mymail.send_message(sender=sender, recipients=recipients, subject=subject, body=body)
        flash("Sent Successfully")
        time.sleep(5)
    return render_template("send_mail.html")

@app.route('/doctory_registration')
def doctory_registration():
    return render_template('doctor_register.html')

#Add Users to the database
def add_doctor_to_database(name, email, country, region, location, place_of_work, password):
    engine.execute("INSERT INTO doctors(name,email,country,region,location,place_of_work,password) VALUES('"+name+"','"+email+"','"+country+"','"+region+"','"+location+"','"+place_of_work+"','"+password+"');")
    engine.execute("INSERT INTO users(name,email,password) VALUES('"+name+"','"+email+"','"+password+"');")

@app.route('/register_doctor', methods = ['POST','GET'])
def register_doctor():
    name = request.form.get('name')
    email = request.form.get('email')
    country = request.form.get('country')
    region = request.form.get('region')
    location = request.form.get('location')
    place_of_work = request.form.get('place_of_work')
    password = request.form.get('password')
    try:
        add_doctor_to_database(name,email,country,region,location,place_of_work,password)
    except:
        return redirect(url_for('doctory_registration', message = "Doctor already exist"))
    return render_template('login.html')

@app.route('/admin')
def admin():
    return render_template('disease_admin.html')