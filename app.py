from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import jwt
import datetime
import joblib
import numpy as np
from functools import wraps
from flask import Flask, session
from flask_session import Session

app = Flask(__name__)


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('JWT_SECRET', 'default_secret_key')
bcrypt = Bcrypt(app)
CORS(app)

# MongoDB connection with modified parameters to handle SSL issues
try:
    mongo_uri = os.getenv('MONGODB_URI')
    print(f"Connecting to MongoDB with URI: {mongo_uri}")
    
    # # Check if mongo_uri is properly set
    # if not mongo_uri or mongo_uri == "mongodb+srv://vehicle:<mypassword>@cluster1.xxuuikb.mongodb.net/vehicle_price_prediction":
    #     raise ValueError("MongoDB URI is not properly configured. Please update your .env file with the correct URI.")
    
    # # Parse the connection string to extract username and password for debugging
    # # Don't print the full URI with credentials in production
    # if "mongodb+srv://" in mongo_uri:
    #     uri_parts = mongo_uri.replace("mongodb+srv://", "").split("@")[0].split(":")
    #     username = uri_parts[0]
    #     # Don't print the actual password, just check if it's set
    #     has_password = len(uri_parts) > 1 and uri_parts[1] != "<mypassword>" and len(uri_parts[1]) > 0
    #     print(f"MongoDB username: {username}, Password set: {has_password}")
    
    # Add SSL and connection parameters to handle SSL handshake issues
    client = MongoClient(
        mongo_uri,
        ssl=True,
        tls=True,  # Add TLS support explicitly
        tlsAllowInvalidCertificates=True,  # Allow invalid certificates for testing
        retryWrites=True,
        w="majority",
        connectTimeoutMS=30000,  # Increase connection timeout
        serverSelectionTimeoutMS=30000,  # Increase server selection timeout
        socketTimeoutMS=30000  # Increase socket timeout
    )
    
    # Test connection
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(f"MongoDB connection error: {str(e)}")
    # Fallback to a local dictionary for development/testing
    print("Using in-memory storage as fallback")
    client = None

# Extract database name from URI or use default
if mongo_uri and 'vehicle' in mongo_uri:
    db_name = 'vehicle_price_prediction'
else:
    # Extract database name from the URI or use default
    db_name = mongo_uri.split('/')[-1] if mongo_uri and '/' in mongo_uri else 'vehicle_price_prediction'

print(f"Using database: {db_name}")

# Setup database and collections
if client:
    db = client[db_name]
    users_collection = db.users
    contacts_collection = db.contacts
else:
    # Fallback to in-memory storage for development/testing
    class InMemoryCollection:
        def __init__(self):
            self.data = []
            self.counter = 0
            
        def insert_one(self, document):
            self.counter += 1
            document['_id'] = str(self.counter)
            self.data.append(document)
            class Result:
                def __init__(self, id):
                    self.inserted_id = id
            return Result(document['_id'])
            
        def find_one(self, query):
            for doc in self.data:
                match = True
                for key, value in query.items():
                    if key not in doc or doc[key] != value:
                        match = False
                        break
                if match:
                    return doc
            return None
            
        def count_documents(self, query):
            count = 0
            for doc in self.data:
                match = True
                for key, value in query.items():
                    if key not in doc or doc[key] != value:
                        match = False
                        break
                if match:
                    count += 1
            return count
    
    class InMemoryDB:
        def __init__(self):
            self.users = InMemoryCollection()
            self.contacts = InMemoryCollection()
            
        def __getitem__(self, name):
            if name == 'users':
                return self.users
            elif name == 'contacts':
                return self.contacts
            raise KeyError(f"Collection {name} not found")
            
        def list_collection_names(self):
            return ['users', 'contacts']
    
    db = InMemoryDB()
    users_collection = db.users
    contacts_collection = db.contacts

# Load the model
try:
    model = joblib.load('vehicle_price_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Update the label encoders with actual values from your dataset
# These mappings should match what was used during model training
make_mapping = {
    'Toyota': 0, 'Honda': 1, 'Ford': 2, 'Chevrolet': 3, 'BMW': 4, 
    'Mercedes-Benz': 5, 'Audi': 6, 'Nissan': 7, 'Volkswagen': 8, 'Hyundai': 9
}

model_mapping = {
    'Camry': 0, 'Corolla': 1, 'RAV4': 2, 'Civic': 3, 'Accord': 4, 
    'CR-V': 5, 'F-150': 6, 'Escape': 7, 'Silverado': 8, '3 Series': 9,
    'C-Class': 10, 'A4': 11, 'Altima': 12, 'Jetta': 13, 'Elantra': 14
}

engine_mapping = {
    '1.0L': 0, '1.5L': 1, '2.0L': 2, '2.5L': 3, '3.0L': 4, 
    '3.5L': 5, '4.0L': 6, '5.0L': 7
}

fuel_mapping = {
    'Gasoline': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3, 'Plug-in Hybrid': 4
}

transmission_mapping = {
    'Automatic': 0, 'Manual': 1, 'CVT': 2, 'DCT': 3
}

body_mapping = {
    'Sedan': 0, 'SUV': 1, 'Hatchback': 2, 'Coupe': 3, 'Truck': 4,
    'Van': 5, 'Wagon': 6, 'Convertible': 7
}

drivetrain_mapping = {
    'FWD': 0, 'RWD': 1, 'AWD': 2, '4WD': 3
}

# Token required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token and 'token' in session:
            token = session['token']
            
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            current_user = users_collection.find_one({'_id': data['user_id']})
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated

# Check if user is logged in
def is_logged_in():
    return 'token' in session

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict-page')
def predict_page():
    # Check if user is logged in
    if not is_logged_in():
        # Store the intended destination for redirect after login
        session['next_url'] = url_for('predict_page')
        # Flash a message to inform the user
        return redirect(url_for('login_page'))
    return render_template('prediction.html')

@app.route('/contact')
def contact():
    # Check if user is logged in
    if not is_logged_in():
        # Store the intended destination for redirect after login
        session['next_url'] = url_for('contact')
        # Redirect to login page
        return redirect(url_for('login_page'))
    return render_template('contact.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.form
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': data['email']})
        if existing_user:
            return jsonify({'message': 'User already exists!'}), 400
        
        # Hash the password
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        
        # Create new user
        new_user = {
            'name': data['name'],
            'email': data['email'],
            'password': hashed_password,
            'created_at': datetime.datetime.utcnow()
        }
        
        # Insert user into database
        result = users_collection.insert_one(new_user)
        print(f"User registered with ID: {result.inserted_id}")
        
        return jsonify({'message': 'User registered successfully!'}), 201
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.form
        
        # Find user in database
        user = users_collection.find_one({'email': data['email']})
        
        if not user or not bcrypt.check_password_hash(user['password'], data['password']):
            return jsonify({'message': 'Invalid credentials!'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.secret_key, algorithm="HS256")
        
        # Store token in session
        session['token'] = token
        
        # Check if there's a next_url to redirect to after login
        next_url = session.pop('next_url', None)
        
        # Return the next_url along with the token
        return jsonify({
            'token': token,
            'next_url': next_url or url_for('predict_page')
        }), 200
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'message': f'Login failed: {str(e)}'}), 500

@app.route('/logout')
def logout():
    session.pop('token', None)
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@token_required  # Enable token protection
def predict(current_user):
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
            
        data = request.form
        print(f"Received prediction data: {data}")
        
        # Get values from form
        make = data.get('make', '')
        model_name = data.get('model', '')
        year = int(data.get('year', 2020))
        mileage = float(data.get('mileage', 0))
        engine = data.get('engine', '')
        fuel = data.get('fuel', '')
        transmission = data.get('transmission', '')
        body = data.get('body', '')
        drivetrain = data.get('drivetrain', '')
        cylinders = float(data.get('cylinders', 4))
        doors = float(data.get('doors', 4))
        
        # Simple mapping for demonstration - in production, these should match your training data
        # These are placeholder values - you should replace with actual mappings from your model training
        make_idx = {'Toyota': 0, 'Honda': 1, 'Ford': 2, 'Chevrolet': 3, 'BMW': 4, 
                   'Mercedes-Benz': 5, 'Audi': 6, 'Nissan': 7, 'Volkswagen': 8, 'Hyundai': 9}.get(make, 0)
        
        model_idx = {'Camry': 0, 'Corolla': 1, 'RAV4': 2, 'Civic': 3, 'Accord': 4, 
                    'CR-V': 5, 'F-150': 6, 'Escape': 7, 'Silverado': 8, '3 Series': 9}.get(model_name, 0)
        
        engine_idx = {'1.0L': 0, '1.5L': 1, '2.0L': 2, '2.5L': 3, '3.0L': 4, 
                     '3.5L': 5, '4.0L': 6, '5.0L': 7}.get(engine, 0)
        
        fuel_idx = {'Gasoline': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3, 'Plug-in Hybrid': 4}.get(fuel, 0)
        
        transmission_idx = {'Automatic': 0, 'Manual': 1, 'CVT': 2, 'DCT': 3}.get(transmission, 0)
        
        body_idx = {'Sedan': 0, 'SUV': 1, 'Hatchback': 2, 'Coupe': 3, 'Truck': 4,
                   'Van': 5, 'Wagon': 6, 'Convertible': 7}.get(body, 0)
        
        drivetrain_idx = {'FWD': 0, 'RWD': 1, 'AWD': 2, '4WD': 3}.get(drivetrain, 0)
        
        # Create feature array - IMPORTANT: order must match training data
        # Based on your get_vehicle_price function, the order appears to be:
        # [make, model, year, mileage, engine, fuel, transmission, body, drivetrain, cylinders, doors]
        features = np.array([[
            make_idx, model_idx, year, mileage, engine_idx, 
            fuel_idx, transmission_idx, body_idx, drivetrain_idx, 
            cylinders, doors
        ]])
        
        print(f"Prediction input features: {features}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        print(f"Raw prediction: {prediction}")
        
        # Format prediction
        formatted_prediction = float(prediction)
        
        return jsonify({'prediction': formatted_prediction}), 200
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/contact", methods=["POST"])
# @token_required
def submit_contact():
    try:
        # Check if data is coming as JSON
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()  # Convert ImmutableMultiDict to dict

        print(f"Received Data: {data}")  # Debugging

        # Validate required fields
        required_fields = ["name", "email", "message"]
        if not all(field in data for field in required_fields):
            return jsonify({"message": "Missing required fields"}), 400

        # Insert into MongoDB
        contact_entry = {
            "name": data["name"],
            "email": data["email"],
            "subject": data.get("subject", ""),  # Optional
            "message": data["message"]
        }
        result = contacts_collection.insert_one(contact_entry)

        print(f"Inserted Contact ID: {result.inserted_id}")  # Debugging

        return jsonify({"message": "Form submitted successfully!"}), 201

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500



@app.route('/test-model', methods=['GET'])
def test_model():
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'model_loaded': False,
                'error': 'Model not loaded',
                'message': 'Error loading model file'
            }), 500
            
        # Create a sample input with default values - using the same format as in your notebook
        # Based on your get_vehicle_price function: vehicle_data = np.array([[15, 54, 2024, 65, 6.0, 4, 1.0, 19, 6, 4.0, 1]])
        sample_input = np.array([[15, 54, 2024, 65, 6.0, 4, 1.0, 19, 6, 4.0, 1]])
        
        # Make prediction
        prediction = model.predict(sample_input)[0]
        
        return jsonify({
            'status': 'success',
            'model_loaded': True,
            'sample_prediction': float(prediction),
            'message': 'Model is working correctly',
            'sample_input': sample_input.tolist()
        }), 200
    except Exception as e:
        print(f"Test model error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'model_loaded': model is not None,
            'error': str(e),
            'message': 'Error testing model'
        }), 500

@app.route('/check-db', methods=['GET'])
def check_db():
    try:
        # Check MongoDB connection
        if client:
            db_names = client.list_database_names()
        else:
            db_names = ["in-memory-db"]
            
        collections = db.list_collection_names()
        
        # Count documents in collections
        users_count = users_collection.count_documents({})
        contacts_count = contacts_collection.count_documents({})
        
        return jsonify({
            'status': 'success',
            'connected': client is not None,
            'databases': db_names,
            'current_db': db_name,
            'collections': collections,
            'users_count': users_count,
            'contacts_count': contacts_count
        }), 200
    except Exception as e:
        print(f"Database check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'connected': False,
            'error': str(e),
            'message': 'Error connecting to database'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

