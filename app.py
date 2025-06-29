import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU usage
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import uuid
import tensorflow.lite as tflite
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField, DecimalField, FloatField
from wtforms.validators import DataRequired, EqualTo, Length, NumberRange, Email
from PIL import Image
from tensorflow.keras.preprocessing import image  # type: ignore
from flask_babel import Babel, _
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_mail import Mail, Message
from os import environ
from groq import Groq
import time
import psutil
import threading
from flask_caching import Cache
import requests
import tensorflow as tf
from datetime import datetime, timedelta

app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = environ.get('SECRET_KEY', 'your_secret_key')  # Ensure a default for safety
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit uploads to 1 MB
app.config['MAIL_SERVER'] = environ.get('MAIL_SERVER')
app.config['MAIL_PORT'] = environ.get('MAIL_PORT')
app.config['MAIL_USERNAME'] = environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = environ.get('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
app.config['CACHE_TYPE'] = 'SimpleCache'  # Use simple in-memory cache
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('SQLALCHEMY_DATABASE_URI')
print(f"SQLALCHEMY_DATABASE_URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

print(f"MAIL_SERVER: {app.config['MAIL_SERVER']}")
print(f"MAIL_PORT: {app.config['MAIL_PORT']}")
print(f"MAIL_USERNAME: {app.config['MAIL_USERNAME']}")
print(f"MAIL_PASSWORD: {app.config['MAIL_PASSWORD']}")

# Log TensorFlow device placement
print("TensorFlow devices:", tf.config.list_physical_devices())
if not tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using CPU only")
else:
    print("Warning: TensorFlow is still trying to use GPU")

# Debug CA certificate file
ca_file_path = 'ca.pem'
print(f"CA file exists: {os.path.exists(ca_file_path)}")
print(f"CA file path: {os.path.abspath(ca_file_path)}")
if os.path.exists(ca_file_path):
    print(f"CA file size: {os.path.getsize(ca_file_path)} bytes")
else:
    print("CA file not found!")

# Ensure the uploads directory exists
uploads_dir = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'])
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
    print(f"Created uploads directory: {uploads_dir}")
else:
    print(f"Uploads directory already exists: {uploads_dir}")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
cache = Cache(app)

babel = Babel(app)

def get_locale():
    return session.get('lang', 'en')

babel = Babel(app, locale_selector=get_locale)

@app.route('/set_language/<language>')
@login_required  # Ensure only logged-in users can change language
def set_language(language):
    # Preserve the user ID in the session to prevent logout
    user_id = session.get('_user_id')
    session['lang'] = language
    # Ensure the user ID is retained after session modification
    if user_id:
        session['_user_id'] = user_id
    # Redirect to the referrer if available, otherwise to the dashboard
    referrer = request.referrer if request.referrer else url_for('dashboard')
    return redirect(referrer)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'user'
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

# Load TensorFlow Lite model
print("Loading TensorFlow Lite model...")
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024
print(f"Memory before loading model: {mem_before:.2f} MB")

interpreter = tflite.Interpreter(model_path='desnet2_optimized.tflite')
interpreter.allocate_tensors()
print("TensorFlow Lite model loaded successfully")

mem_after = process.memory_info().rss / 1024 / 1024
print(f"Memory after loading model: {mem_after:.2f} MB")
print(f"Memory increase: {mem_after - mem_before:.2f} MB")

classes = ['American Bollworm on Cotton', 'Anthracnose on Cotton', 'Aphids', 'Army worm', 'Carpetweeds', 'Crabgrass', 'Eclipta',
'Flag Smut', 'Goosegrass', 'Healthy', 'Leaf Curl', 'Leaf smut', 'Morningglory', 'Mosaic sugarcane', 'Nutsedge', 'PalmerAmaranth',
'Powdery_Mildew', 'Prickly Sida', 'Purslane', 'Ragweed', 'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sicklepod',
'SpottedSpurge', 'SpurredAnoda', 'Sugarcane Healthy', 'Swinecress', 'Target_spot', 'Tungro', 'Waterhemp', 'Wheat Brown leaf Rust',
'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew', 'Wheat scab',
'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane', 'bacterial_blight in Cotton', 'bollrot on Cotton', 'bollworm on Cotton',
'cotton mealy bug', 'cotton whitefly', 'curl_virus', 'fussarium_wilt', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
'pink bollworm in cotton', 'red cotton bug', 'thirps on cotton', 'Other']

def predict_disease(image_path, interpreter, threshold=0.045):
    print("Starting prediction in thread...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"Memory before prediction: {mem_before:.2f} MB")

    test_image = image.load_img(image_path, target_size=(256, 256))
    plt.imshow(plt.imread(image_path))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], test_image)

    # Run inference
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    probabilities = tf.nn.softmax(prediction).numpy()

    mem_after = process.memory_info().rss / 1024 / 1024
    print(f"Memory after prediction: {mem_after:.2f} MB")
    print(f"Memory increase: {mem_after - mem_before:.2f} MB")

    max_prob = np.max(probabilities)
    print(f"Max probability: {max_prob}")
    if max_prob < threshold:
        return "Unknown/Not Cotton"
    index = np.argmax(probabilities)
    return classes[index]

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
            msg = Message('Password Reset Request', sender='codcanva@gmail.com', recipients=[email])
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

def run_prediction(file_path, interpreter, threshold=0.045):
    print("Starting prediction in thread...")
    start_time = time.time()
    preds = predict_disease(file_path, interpreter, threshold)
    print(f"Prediction completed in thread (took {time.time() - start_time:.2f} seconds)")
    return preds

@app.route('/disease_detection', methods=['GET', 'POST'])
@login_required
def disease_detection():
    if request.method == 'POST':
        print("Starting disease detection...")
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        print(f"Memory before processing: {mem_before:.2f} MB")

        if 'file' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
            try:
                print("Saving file...")
                start_time = time.time()
                f.save(file_path)
                print(f"File saved successfully: {file_path} (took {time.time() - start_time:.2f} seconds)")
            except Exception as e:
                print(f"Error saving file: {e}")
                flash('Error saving file', 'danger')
                return redirect(request.url)

            mem_after_save = process.memory_info().rss / 1024 / 1024
            print(f"Memory after file saving: {mem_after_save:.2f} MB")
            print(f"Memory increase after saving: {mem_after_save - mem_before:.2f} MB")

            print("Starting prediction thread...")
            result = [None]
            exception = [None]
            def prediction_wrapper():
                try:
                    result[0] = run_prediction(file_path, interpreter, 0.045)
                except Exception as e:
                    exception[0] = e

            prediction_thread = threading.Thread(target=prediction_wrapper)
            prediction_thread.start()
            prediction_thread.join(timeout=60)

            if prediction_thread.is_alive():
                print("Prediction timed out after 60 seconds")
                flash('Prediction timed out', 'danger')
                return redirect(request.url)
            if exception[0]:
                print(f"Prediction error: {exception[0]}")
                flash('Error during prediction', 'danger')
                return redirect(request.url)

            mem_after = process.memory_info().rss / 1024 / 1024
            print(f"Memory after processing: {mem_after:.2f} MB")
            print(f"Memory increase: {mem_after - mem_before:.2f} MB")

            img_url = url_for('static', filename=f'uploads/{secure_filename(f.filename)}')
            return render_template('disease_detection.html', result=result[0], img_url=img_url)
        else:
            flash('Invalid file type. Allowed types: png, jpg, jpeg', 'danger')
            return redirect(request.url)
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
    print("Starting chatbot response generation...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"Memory before API call: {mem_before:.2f} MB")

    user_input = request.form['user_input']
    print(f"User input: {user_input}")

    try:
        start_time = time.time()
        response = generate_response(user_input)
        print(f"Chatbot response: {response} (took {time.time() - start_time:.2f} seconds)")

        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory after API call: {mem_after:.2f} MB")
        print(f"Memory increase: {mem_after - mem_before:.2f} MB")

        return jsonify(response=response)
    except Exception as e:
        print(f"Chatbot error: {e}")
        return jsonify(response="Sorry, I encountered an error. Please try again."), 500

def generate_response(user_input):
    try:
        os.environ['GROQ_API_KEY'] = environ.get('GROQ_API')
        groq_client = Groq()

        messages = [
            {"role": "system", "content": "You are a specialized assistant with expertise in cotton farming. Only provide responses related to cotton farming and crop management. If the user asks about other topics, politely redirect them to cotton farming or say you can only assist with cotton-related questions."},
            {"role": "user", "content": user_input}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=200,
            stream=False,
            timeout=10,
            extra_headers={"X-Request-ID": str(uuid.uuid4())}
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

@app.route('/fertilizer_calculator', methods=['GET', 'POST'])
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
        os.environ['GROQ_API_KEY'] = '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'
        groq_client = Groq()

        messages = [
            {"role": "user", "content": user_input}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",  # Use a lighter model
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
        "date": "2025-05-19"
    },
    {
        "title": "Bacterial Blight Outbreak",
        "description": "Bacterial Blight outbreak reported in several cotton farms. Farmers are advised to monitor their crops closely.",
        "date": "2025-05-18"
    },
    {
        "title": "Fusarium Wilt Warning",
        "description": "Fusarium Wilt has been identified in the northern cotton-growing regions. Preventive measures recommended.",
        "date": "2025-05-16"
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

    # Add sorting and date range to fetch recent news
    url = f"https://newsapi.org/v2/everything?q=cotton AND farmers AND agriculture  -coffee -Nagaland -subsidy -Trump -restaurant -brunch -Bayer -fashion -clothes -clothing -textiles&language=en&sortBy=relevancy&apiKey={api_key}"


    # Add headers to prevent caching
    headers = {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
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
        'access_key': 'your_api_key',
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
    pass  # Do not run the development server in production
