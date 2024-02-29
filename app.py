from flask import Flask,render_template,request, flash, redirect, url_for, session, request, logging,jsonify, json, send_file
from flask_session import Session
from datetime import timedelta
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from flask_mysqldb import MySQL
from passlib.hash import sha256_crypt
from functools import wraps
import pickle
import pandas as pd
from os import environ
import io
import gzip
import joblib
import selenium
from selenium import webdriver,common
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys  import Keys
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.support.wait import WebDriverWait

from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
#importing tine Libraries to add wait times
from datetime import datetime
from time import sleep
#importing beautiful soup to read the page html source code
from bs4 import BeautifulSoup
#to create csv file where we'll scrape the content 
import pandas as pd



app = Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sales_flask'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MYSQL
mysql = MySQL(app)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)

# The maximum number of items the session stores 
# before it starts deleting some, default 500
app.config['SESSION_FILE_THRESHOLD'] = 50  
Session(app)


# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if session['role']=="Admin":
            return f(*args, **kwargs)
        else:
            flash('You need to be an admin to view this page.','warning' )
            return redirect(url_for('home'))

    return wrap    

@app.route("/home",methods=['GET', 'POST'])
@is_logged_in
def home():
    
    
    return render_template("home.html")

@app.route("/dashboard",methods=['GET', 'POST'])
@is_logged_in
def dashboard():
    
    
    return render_template("dashboard.html")

@app.route("/predict",methods=['GET', 'POST'])
@is_logged_in
def predict():
    
    with open('model/model3.pkl', 'rb') as file:
        
        data = joblib.load(file)

    regressor_loaded = data["model"]
 
    le_ROLE_CODE = data["le_ROLE_CODE"]
    #le_PERIODACTIVEDAY = data["le_PERIODACTIVEDAY"]
    le_PRODUCT_CODE = data["le_PRODUCT_CODE"]   
    if request.method == 'POST':
        ROLE_CODE = request.form.get('ROLE_CODE')
        PRODUCT_CODE = request.form.get('PRODUCT_CODE')
        year = request.form.get('year')
        month = request.form.get('month')
        day = request.form.get('day')
        #PERIODACTIVEDAY = request.form.get('PERIODACTIVEDAY')
        
        input_variables = pd.DataFrame([[ROLE_CODE,PRODUCT_CODE,year,month,day ]],
                                       columns=['ROLE_CODE','PRODUCT_CODE','year','month','day']) 

        input_variables['ROLE_CODE'] = le_ROLE_CODE.transform(input_variables['ROLE_CODE'])
        #input_variables['PERIODACTIVEDAY'] = le_PERIODACTIVEDAY.transform(input_variables['PERIODACTIVEDAY'])
        input_variables['PRODUCT_CODE'] = le_PRODUCT_CODE.transform(input_variables['PRODUCT_CODE'])
        prediction = regressor_loaded.predict(input_variables)[0]
        
        return render_template("predict.html", original_input={'ROLE_CODE':ROLE_CODE,

                                                      'PRODUCT_CODE':PRODUCT_CODE,
   
                                                     #'PERIODACTIVEDAY':PERIODACTIVEDAY,
                                                     },

                                     result=int(prediction),)
    return render_template("predict.html")

# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.Length(min=6),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        password = sha256_crypt.encrypt(str(form.password.data))

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        
        cur.execute("SELECT * FROM sales WHERE name=%s", (name,))
        sales = cur.fetchone()
        if sales is not None:
            war = "This name is used"
            return render_template('register.html', form=form, error=war)
        else:    
            cur.execute("INSERT INTO sales(name, email, password) VALUES(%s, %s, %s)", (name, email, password))
            mysql.connection.commit()

            # Close connection
            cur.close()

            flash('You are now registered and can log in', 'success')

            return redirect(url_for('login')) 
    return render_template('register.html', form=form)

# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        name = request.form['name']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by name
        result = cur.execute("SELECT * FROM sales WHERE name = %s", [name])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']
            role = data['role']
            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['name'] = name
                session['role'] = role

                
                return redirect(url_for('home'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'name not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap



# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))    


@app.route('/index')
@is_logged_in
@admin_required
def Index():
    cur = mysql.connection.cursor()
 
    cur.execute('SELECT * FROM sales')
    data = cur.fetchall()
  
    cur.close()
    return render_template('index.html', sales = data)


@app.route('/add_user', methods=['POST'])
@is_logged_in
@admin_required

def add_user():
    
    cur = mysql.connection.cursor()
    
    
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        role = request.form['role']
        password = sha256_crypt.encrypt(str(request.form['password']))

        cur.execute("SELECT * FROM sales WHERE name=%s", (name,))
        sales = cur.fetchone()
        if sales is not None:
            war = "This name is used"
            return render_template('index.html', error=war)
        else:    
            cur.execute("INSERT INTO sales(name, email, password,role) VALUES(%s, %s, %s,%s)", (name, email, password,role))
            mysql.connection.commit()

            # Close connection
            cur.close()

            flash('You are now registered and can log in', 'success')  
            return redirect(url_for('Index'))
    

@app.route('/edit/<id>', methods = ['POST', 'GET'])
@is_logged_in
@admin_required
def get_employee(id):
    
    cur = mysql.connection.cursor()
  
    cur.execute("SELECT * FROM sales WHERE id=%s",(id,))
    data = cur.fetchall()
    cur.close()
    print(data[0])
    return render_template('edit.html', sales = data[0])    
    

@app.route('/update/<id>', methods=['POST'])
@is_logged_in
@admin_required
def update_employee(id):
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        role = request.form['role']
        
        
        cur = mysql.connection.cursor()
        cur.execute("""
            UPDATE sales
            SET name = %s,
                email = %s,
                role= %s  
            WHERE id = %s
        """, (name, email,role, id))
        flash('user Updated Successfully', 'success')
        mysql.connection.commit()
        return redirect(url_for('Index'))


@app.route('/delete/<string:id>', methods = ['POST','GET'])
@is_logged_in
@admin_required
def delete_employee(id):
    
    cur = mysql.connection.cursor()
  
    cur.execute('DELETE FROM sales WHERE id = {0}'.format(id))
    mysql.connection.commit()
    flash('User Removed Successfully', 'success')
    return redirect(url_for('Index'))
   

@app.route("/ajaxfile",methods=["POST","GET"])
def ajaxfile():
    try:
        
        cur = mysql.connection.cursor()
        if request.method == 'POST':
            draw = request.form['draw'] 
            row = int(request.form['start'])
            rowperpage = int(request.form['length'])
            searchValue = request.form["search[value]"]
            print(draw)
            print(row)
            print(rowperpage)
            print(searchValue)
 
            ## Total number of records without filtering
            cur.execute("select count(*) as allcount from sales")
            rsallcount = cur.fetchone()
            totalRecords = rsallcount['allcount']
            print(totalRecords) 
 
            ## Total number of records with filtering
            likeString = "%" + searchValue +"%"
            cur.execute("SELECT count(*) as allcount from sales WHERE name LIKE %s OR email LIKE %s OR role LIKE %s", (likeString, likeString, likeString))
            rsallcount = cur.fetchone()
            totalRecordwithFilter = rsallcount['allcount']
            print(totalRecordwithFilter) 
 
            ## Fetch records
            if searchValue=='':
                cur.execute("SELECT * FROM sales ORDER BY name asc limit %s, %s;", (row, rowperpage))
                employeelist = cur.fetchall()
            else:        
                cur.execute("SELECT * FROM sales WHERE name LIKE %s OR email LIKE %s OR role LIKE %s limit %s, %s;", (likeString, likeString, likeString, row, rowperpage))
                employeelist = cur.fetchall()
 
            data = []
            for row in employeelist:
                data.append({
                    'name': row['name'],
                    'email': row['email'],
                    'role': row['role'],
                    
                })
 
            response = {
                'draw': draw,
                'iTotalRecords': totalRecords,
                'iTotalDisplayRecords': totalRecordwithFilter,
                'aaData': data,
            }
            return jsonify(response)
    except Exception as e:
        print(e)






@app.route("/scrape",methods=['GET', 'POST'])
@is_logged_in
def scrape():
    #add the options functionality to disable notifications 
    chrome_options = Options() 
    #disable notifications 
    chrome_options.add_argument("--disable-notifications") 
    driver= webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)      
    if request.method == 'POST':
        emaill = request.form.get('emaill')
        mdp = request.form.get('mdp')
        url = request.form.get('url')
        
        
        
        driver.get("https://www.facebook.com")
        driver.maximize_window()
        sleep(2) 

        email=driver.find_element_by_id("email")
        email.send_keys(emaill) 
        password=driver.find_element_by_id("pass")
        password.send_keys(mdp) 
        sleep(1)

        login=driver.find_element_by_name("login")
        login.click()
        sleep(2)

        driver.get(url)
        sleep(4)


        soup=BeautifulSoup(driver.page_source,"html.parser") 

        divall=driver.find_element(By.XPATH,"//div[@class='oajrlxb2 g5ia77u1 qu0x051f esr5mh6w e9989ue4 r7d6kgcz rq0escxv nhd2j8a9 nc684nl6 p7hjln8o kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x jb3vyjys rz4wbd8a qt6c0cv9 a8nywdso i1ao9s8h esuyzwwr f1sip0of n00je7tq arfg74bv qs9ysxi8 k77z8yql l9j0dhe7 abiwlrkh p8dawk7l lzcic4wl']") 

        if divall.is_displayed():
            driver.execute_script("arguments[0].click();",divall)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Tous les commentaires']"))).click()
        else:
            pass
        sleep(3)
        
        
        try:
            while driver.find_element(By.XPATH,"//div[@class='oajrlxb2 g5ia77u1 mtkw9kbi tlpljxtp qensuy8j ppp5ayq2 goun2846 ccm00jje s44p3ltw mk2mc5f4 rt8b4zig n8ej3o3l agehan2d sk4xxmp2 rq0escxv nhd2j8a9 mg4g778l p7hjln8o kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x tgvbjcpo hpfvmrgz jb3vyjys qt6c0cv9 a8nywdso l9j0dhe7 i1ao9s8h esuyzwwr f1sip0of du4w35lb n00je7tq arfg74bv qs9ysxi8 k77z8yql pq6dq46d btwxx1t3 abiwlrkh lzcic4wl bp9cbjyn m9osqain buofh1pr g5gj957u p8fzw8mz gpro0wi8']").is_displayed():
                driver.execute_script("arguments[0].click();", driver.find_element(By.XPATH,"//div[@class='oajrlxb2 g5ia77u1 mtkw9kbi tlpljxtp qensuy8j ppp5ayq2 goun2846 ccm00jje s44p3ltw mk2mc5f4 rt8b4zig n8ej3o3l agehan2d sk4xxmp2 rq0escxv nhd2j8a9 mg4g778l p7hjln8o kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x tgvbjcpo hpfvmrgz jb3vyjys qt6c0cv9 a8nywdso l9j0dhe7 i1ao9s8h esuyzwwr f1sip0of du4w35lb n00je7tq arfg74bv qs9ysxi8 k77z8yql pq6dq46d btwxx1t3 abiwlrkh lzcic4wl bp9cbjyn m9osqain buofh1pr g5gj957u p8fzw8mz gpro0wi8']"))
                sleep(1)
        except:
            pass

        comment_list=[] 
        time_list=[] 
        name_list=[] 
        soup=BeautifulSoup(driver.page_source,"html.parser") 
        all_com= soup.findAll("div",{"class":"tw6a2znq sj5x9vvc d1544ag0 cxgpxx05"})
        #text=driver.find_element(By.XPATH,"//div[@class='kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql']")
        for com in all_com:
             if com.find("div",{"class":"kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql"}) == None:
                 pass
             else:
                 try:
                     names=com.find("span",{"class":"nc684nl6"})
                 except:
                     names="not found"
                 
                 try:
                     comments=com.find("div",{"class":"kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql"})
        
                 except:
                     comments="not found"
                 
            
    
     
             name_list.append(names.text)
             comment_list.append(comments.text)
      
        import pandas as pd

        df=pd.DataFrame({"name":name_list,"comment":comment_list})
        df.drop_duplicates(subset="comment",keep='first',inplace=True)
        csv = df.to_csv(index=False, header=True, sep=";")    
        dict_obj = df.to_dict('list')
        session['data'] = dict_obj
        
    
        # Create a string buffer
        buf_str = io.StringIO(csv)

        # Create a bytes buffer from the string buffer
        buf_byt = io.BytesIO(buf_str.read().encode("utf-8"))
    
        
        
    
        return send_file(buf_byt,
                     mimetype="text/csv",
                     as_attachment=True,
                     attachment_filename="data.csv")
    return render_template("scrape.html")






@app.route("/download", methods=["POST"])
def download():
    # Get the CSV data as a string from the session
    csv = session["df"] if "df" in session else ""
    
    # Create a string buffer
    buf_str = io.StringIO(csv)

    # Create a bytes buffer from the string buffer
    buf_byt = io.BytesIO(buf_str.read().encode("utf-8"))
    
    # Return the CSV data as an attachment
    return send_file(buf_byt,
                     mimetype="text/csv",
                     as_attachment=True,
                     attachment_filename="data.csv")

@app.route("/word")
@is_logged_in
def word():
    import numpy as np
    from wordcloud import WordCloud 
    from nltk.corpus import stopwords
    from matplotlib import pyplot as plt
 # Get the CSV data as a string from the session
    dict_obj = session['data'] if 'data' in session else ""  
    df = pd.DataFrame(dict_obj)

    import googletrans
    from googletrans import Translator

    translator = Translator()

    def toenglish(x):
        translator = Translator(service_urls=['translate.googleapis.com'])
        result = translator.translate(x, dest='en')
        return result.text

    df['translate_comment'] = list(map(toenglish, df['comment']))
   

    #remove all punctuation like '? ! ; ..'
    import string 
    def remove_punctuation(text):
        txt_nopunct= "".join([c for c in text if c not in string.punctuation])
        return txt_nopunct

    #remove alll urls like http:www.facebook.com
    from nltk.tokenize import TweetTokenizer
    import re
    def remove_urls(text):
        tk=TweetTokenizer() 
        text_tokens = tk.tokenize(text)
        url_pattern = re.compile(r'(https|http)?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)    

    #delete all number in all text in dataset
    def clean_numbers(text):
        tk=TweetTokenizer() 
        text_tokens = tk.tokenize(text)
        # remove numbers
        text_nonum = re.sub(r'\d+', '', text)
        return text_nonum
    
    #remove urls in all message (rows) using apply 
    df['translate_comment'] = df['translate_comment'].apply(remove_urls)
    #remove all number in all message
    df['translate_comment'] = df['translate_comment'].apply(clean_numbers)
    #remove all punctuation like '!,?,; ...'
    df['translate_comment'] = df['translate_comment'].apply(lambda text: remove_punctuation(text))
    df['comment_without_stopwords'] = df['translate_comment'].str.replace(r'I', '')
    df['comment_without_stopwords'] = df['translate_comment'].str.replace(r'the', '')
    #effacer les mots vides 
    from nltk.corpus import stopwords

    
    stop = stopwords.words('english')
    df['translate_comment']= df["translate_comment"].str.lower()
    #df['comment_without_stopwords']=df['comment_without'].apply(lambda x: [item for item in x if item not in stop])
    df['comment_without_stopwords'] = df['translate_comment'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    from autocorrect import Speller

    from nltk.stem.porter import PorterStemmer

    ## Use English stemmer.
    stemmer = PorterStemmer()
    df['stemmed'] = df['comment_without_stopwords'].str.split()
    df['stemmed'] = df['stemmed'].apply(lambda x: [stemmer.stem(y) for y in x])

    text = str(df['stemmed'].values)
    stop = stopwords.words('english')
    wc = WordCloud(width=600,height=600,background_color="white", max_words=200, stopwords=stop,  max_font_size=200,collocations = False, random_state=42)

    # Générer et afficher le nuage de mots
    plt.figure()
    wc.generate(text)
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    
    plt.savefig('static/images/new_plot.jpg')
    plt.close()

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    import plotly.graph_objs as go
    from plotly.utils import PlotlyJSONEncoder
    import plotly
 
    import json

    SIA = SentimentIntensityAnalyzer()
    df['compound'] = df['comment_without_stopwords'].apply(lambda message: SIA.polarity_scores(message))
    df['compound']  = df['compound'].apply(lambda score_dict: score_dict['compound'])
    df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >0 else ('neg' if c<0 else 'neut'))       
    
    a=df.comp_score.value_counts().pos
    b=df.comp_score.value_counts().neg
    c=df.comp_score.value_counts().neut

    sent = [a, b, c]
    nsent=['pos','neg', 'neut']
    trace1 = go.Bar(x=['pos','neg', 'neut'], y=[a, b, c],marker=dict(color=[ '#2ca02c','#d62728','#7f7f7f'],
                     colorscale='viridis'),width=0.4)
    layout = go.Layout(title="Sentiment analysis", xaxis=dict(title="Sentiment"),
                       yaxis=dict(title="number of users "), )
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
 
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    
    
    return render_template("plot.html",url ='static/images/new_plot.jpg',plot=fig_json)

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == "__main__":
    app.secret_key='houssem123'
    
    app.run(debug=True)
