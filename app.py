import os
import requests
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField, DecimalField, FloatField
from wtforms.validators import DataRequired, EqualTo, Length, NumberRange
from wtforms.validators import DataRequired, Email, EqualTo
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from flask_babel import Babel, _
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_mail import Mail, Message
from os import environ
from groq import Groq
import time

app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = environ.get('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Added to suppress warning
app.config['MAIL_SERVER'] = environ.get('MAIL_SERVER')
app.config['MAIL_PORT'] = environ.get('MAIL_PORT')
app.config['MAIL_USERNAME'] = environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = environ.get('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

db = SQLAlchemy(app)
mail = Mail(app)
s = URLSafeTimedSerializer('your_secret_key')

babel = Babel(app)

def get_locale():
    return session.get('lang', 'en')

babel = Babel(app, locale_selector=get_locale)

@app.route('/set_language/<language>')
def set_language(language):
    session['lang'] = language
    return redirect(request.referrer)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'user'  # Match the table name in cotton.sql
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Submit')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


MODEL_PATH ='desnet2.h5'

# load trained model
model = load_model(MODEL_PATH)

# model =load_model("DenseNet121.h5")
classes = ['American Bollworm on Cotton', 'Anthracnose on Cotton', 'Aphids', 'Army worm', 'Carpetweeds', 'Crabgrass', 'Eclipta',
'Flag Smut', 'Goosegrass', 'Healthy', 'Leaf Curl', 'Leaf smut', 'Morningglory', 'Mosaic sugarcane', 'Nutsedge', 'PalmerAmaranth',
'Powdery_Mildew', 'Prickly Sida', 'Purslane', 'Ragweed', 'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sicklepod',
'SpottedSpurge', 'SpurredAnoda', 'Sugarcane Healthy', 'Swinecress', 'Target_spot', 'Tungro', 'Waterhemp', 'Wheat Brown leaf Rust',
'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew', 'Wheat scab',
'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane', 'bacterial_blight in Cotton', 'bollrot on Cotton', 'bollworm on Cotton',
'cotton mealy bug', 'cotton whitefly', 'curl_virus', 'fussarium_wilt', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
'pink bollworm in cotton', 'red cotton bug', 'thirps on  cotton','Other']


def predict_disease(image_path,model,threshold = 0.045):
    
    test_image = image.load_img(image_path,target_size = (256,256))
    plt.imshow(plt.imread(image_path))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)[0]
    probabilities = tf.nn.softmax(prediction).numpy()
    # result = model.predict(test_image)
    # result = result.ravel() 
    # max = result[0];
    max_prob = np.max(probabilities)
    print(max_prob)
    if max_prob <threshold:
        return "Unknown/Not Cotton"
    # index = 0; 
    index = np.argmax(probabilities)
    #Loop through the array    
    # for i in range(0, len(result)):    
    #   #Compare elements of array with max    
    #   if(result[i] > max):    
    #       max = result[i];    
    #       index = i
    #print("Largest element present in given array: " + str(max) +" And it belongs to " +str(classes[index]) +" class."); 
    # pred = str(classes[index])
    return classes[index]


# def model_predict(img_path, model):
#     print('Uploaded image path: ',img_path)
#     loaded_image = image.load_img(img_path, target_size=(224, 224))

#     # preprocess the image
#     loaded_image_in_array = image.img_to_array(loaded_image)

#     # normalize
#     loaded_image_in_array=loaded_image_in_array/255

#     # add additional dim such as to match input dim of the model architecture
#     x = np.expand_dims(loaded_image_in_array, axis=0)

#     # prediction
#     prediction = model.predict(x)

#     results=np.argmax(prediction, axis=1)

#     if results==0:
#         results="The leaf is diseased cotton leaf"
#     elif results==1:
#         results="The leaf is diseased cotton plant"
#     elif results==2:
#         results="The leaf is fresh cotton leaf"
#     else:
#         results="The leaf is fresh cotton plant"

#     return results




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            flash('Logged in successfully!', 'success')
            return render_template('dashboard.html', username=user.username)
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        if user:
            token = s.dumps(email, salt='email-confirm')
            msg = Message('Password Reset Request', sender='your_email@gmail.com', recipients=[email])
            link = url_for('reset_password', token=token, _external=True)
            msg.body = f'Your link is {link}'
            mail.send(msg)
            flash('Check your email for the password reset link')
        else:
            flash('Email not found')
    return render_template('forgot_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=3600)
    except SignatureExpired:
        flash('The reset link has expired.', 'danger')
        return redirect(url_for('forgot_password'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        password = form.password.data
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = password
            db.session.commit()
            flash('Your password has been updated!', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found.', 'danger')
            return redirect(url_for('forgot_password'))
    return render_template('reset_password.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)


@app.route('/disease_detection', methods=['GET', 'POST'])
@login_required
def disease_detection():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_disease(file_path, model,0.045)
        result = preds
        img_url = url_for('static', filename=f'uploads/{secure_filename(f.filename)}')
        result=preds
        return render_template('disease_detection.html',result=result,img_url=img_url)
    return render_template('disease_detection.html')

@app.route('/weather_forecasting', methods=['GET', 'POST'])
@login_required
def weather_forecasting():
    api_key = environ.get('Weather_API_KEY')
    city = "Lahore"
    
    if request.method == 'POST':
        city = request.form['city']
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if data["cod"] != "404":
        weather_data = {
            "city": city,
            "temperature": data["main"]["temp"],
            "pressure": data["main"]["pressure"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"],
        }
    else:
        weather_data = None

    return render_template('weather_forecasting.html', weather_data=weather_data)

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')


@app.route('/get_response', methods=['POST'])
@login_required
def get_response():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return jsonify(response=response)

def generate_response(user_input):
    try:
        # Call the OpenAI API to get a response
        os.environ['GROQ_API_KEY'] = environ.get('GROQ_API')
        groq_client = Groq()

        messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in farming and crop management."},
                
            ]

        # human = input('USER: ')
        messages.append({"role":"user",
                    "content":user_input})
        a = time.time()

        chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192", #mixtral-8x7b-32768 #llama3-8b-8192 #llama3-70b-8192 #gpt-3.5-turbo-0125
        temperature=0.1,
        max_tokens=200,
        stream=False,
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error: {str(e)}"


class FertilizerForm1(FlaskForm):
    area = DecimalField('Area (hectares)', validators=[DataRequired(), NumberRange(min=0.1)], places=2)
    nitrogen = DecimalField('Nitrogen (kg/ha)', validators=[DataRequired(), NumberRange(min=0)], places=2)
    phosphorus = DecimalField('Phosphorus (kg/ha)', validators=[DataRequired(), NumberRange(min=0)], places=2)
    potassium = DecimalField('Potassium (kg/ha)', validators=[DataRequired(), NumberRange(min=0)], places=2)
    submit = SubmitField('Calculate')

def calculate_fertilizer(area, nitrogen, phosphorus, potassium):
    total_nitrogen = area * nitrogen
    total_phosphorus = area * phosphorus
    total_potassium = area * potassium
    return total_nitrogen, total_phosphorus, total_potassium

@app.route('/n_fertilizer', methods=['GET', 'POST'])
@login_required
def n_fertilizer():
    form = FertilizerForm1()
    result = None
    if form.validate_on_submit():
        area = form.area.data
        nitrogen = form.nitrogen.data
        phosphorus = form.phosphorus.data
        potassium = form.potassium.data
        total_nitrogen, total_phosphorus, total_potassium = calculate_fertilizer(area, nitrogen, phosphorus, potassium)
        result = {
            'total_nitrogen': total_nitrogen,
            'total_phosphorus': total_phosphorus,
            'total_potassium': total_potassium
        }
    return render_template('fertilizer_calculator_simple.html', form=form, result=result)


class FertilizerForm(FlaskForm):
    nitrogen = FloatField('Nitrogen', validators=[DataRequired()])
    phosphorus = FloatField('Phosphorus', validators=[DataRequired()])
    potassium = FloatField('Potassium', validators=[DataRequired()])
    submit = SubmitField('Get Advice')


@app.route('/fertilizer_calculator_1', methods=['GET', 'POST'])
@login_required
def fertilizer_calculator():
    result = ''
    form = FertilizerForm()
    if form.validate_on_submit():
        nitrogen = form.nitrogen.data
        phosphorus = form.phosphorus.data
        potassium = form.potassium.data

        user_input = f'Provide advice on using Nitrogen: {nitrogen} kg, Phosphorus: {phosphorus} kg, Potassium: {potassium} kg for cotton fields.'
        response = generate_r(user_input)
        
        result = response
        return render_template('fertilizer_calculator_1.html', form=form, result=result)

    return render_template('fertilizer_calculator_1.html', form=form, result=result)

def generate_r(user_input=None):
    if not user_input:
        user_input = request.json.get('user_input')
    try:
        # Call the OpenAI API to get a response
        os.environ['GROQ_API_KEY'] = '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'
        groq_client = Groq()

        messages = [
            {"role": "user", "content": user_input}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",  # Adjust the model as needed
            temperature=0.1,
            max_tokens=500,
            stream=False,
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error: {str(e)}"



    
# Dummy database for disease alerts
disease_alerts = [
    {
        "title": "Leaf Spot Disease Alert",
        "description": "Leaf Spot disease detected in cotton fields in the southern region. Immediate action required.",
        "date": "2025-02-10"
    },
    {
        "title": "Bacterial Blight Outbreak",
        "description": "Bacterial Blight outbreak reported in several cotton farms. Farmers are advised to monitor their crops closely.",
        "date": "2025-02-08"
    },
    {
        "title": "Fusarium Wilt Warning",
        "description": "Fusarium Wilt has been identified in the northern cotton-growing regions. Preventive measures recommended.",
        "date": "2025-02-03"
    }
]


@app.route('/alerts')
@login_required
def alerts():
    return render_template('alerts.html', alerts=disease_alerts)

@app.route('/news')
@login_required
def news():
    api_key = "70181cc2861d4ffbb83a92f38b3f9662"
    url = f"https://newsapi.org/v2/everything?q=cotton-farming&apiKey={api_key}"

    response = requests.get(url)
    data = response.json()

    articles = []
    if data["status"] == "ok":
        articles = data["articles"]

    return render_template('news.html', articles=articles)


@app.route('/cotton_seeds')
@login_required
def cotton_seeds():
    seed_varieties = [
        {"name": "Bt Cotton", "description": "Bt cotton is genetically modified to produce an insecticide to combat bollworm."},
        {"name": "IR Cotton", "description": "IR cotton varieties are known for their resistance to certain pests and diseases."},
        {"name": "Desi Cotton", "description": "Desi cotton varieties are traditional varieties that are well-suited to local growing conditions."},
        {"name": "Hybrid Cotton", "description": "Hybrid cotton varieties are bred to increase yield and improve fiber quality."},
    ]
    return render_template('cotton_seeds.html', seeds=seed_varieties)

def fetch_blog_posts():
    api_url = "http://api.mediastack.com/v1/news"
    params = {
        'access_key': 'your_api_key',  # Replace with your actual Mediastack API key
        'categories': 'agriculture',
        'keywords': 'farming, cotton',
        'languages': 'en',
        'limit': 5
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        blog_posts = [
            {
                "title": article["title"],
                "content": article["description"],
                "url": article["url"]
            }
            for article in data["data"]
        ]
    except requests.RequestException as e:
        print(f"Error fetching blog posts: {e}")
        blog_posts = [
            {"title": "Importance of Crop Rotation", "content": "Crop rotation helps in maintaining soil fertility and reducing soil erosion..."},
            {"title": "Pest Management in Cotton Farming", "content": "Effective pest management practices are crucial for a healthy cotton crop..."},
            {"title": "Organic Cotton Farming", "content": "Organic cotton farming avoids the use of synthetic chemicals..."},
        ]
    
    return blog_posts

@app.route('/blog')
@login_required
def blog():
    blog_posts = fetch_blog_posts()
    return render_template('blog.html', posts=blog_posts)



if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
