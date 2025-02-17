# Flask Backend for Coral Analysis

## 📌 Overview
This repository contains a Flask backend for coral analysis, integrating **Segment Anything Model (SAM)** for image segmentation. It follows a modular structure with clear separation between **routes, services, models, and schemas** to ensure scalability and maintainability.

## 🏗️ Project Structure
```
app/
│── data             # Data folder. Saves images and CT Scans, and all other kind of data
│   |── meso-scale   # Coral images 
│   |── micro-scale  # Coral CT Scans
│   |── *            # Other data
│── database         # Database models (SQLAlchemy). Define the database scheme
│   |── coral.py     
│── schemas          # API request/response validation (Marshmallow). Define the data format for requests/responses
│   │── *.py         
│── routes/          # API endpoints (Blueprints)
│   │── *.py         
│── services/        # Defines actual functionalities. Disconnected from routes.
│   │── *.py  
│── __init__.py      # Flask app factory function. Builds the app
config.py            # Configuration settings (database, environment variables, paths etc.)
migrations/          # Database migration scripts (Flask-Migrate) = Source control for db
.env                 # Environment variables (e.g., DB credentials)
requirements.txt     # Project dependencies
Readme.md            # Project documentation
run.py               # Entry point to start the Flask application
```

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone <repo-url>
cd <repo-folder>
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a **.env** file in the project root and define required environment variables:
```ini
FLASK_ENV=development
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///database.db
UPLOAD_FOLDER=uploads/
```

### 5️⃣ Initialize the Database
```sh
flask db init
flask db migrate -m "Initial migration."
flask db upgrade
```

### 6️⃣ Run the Flask Application
```sh
python run.py
```
Server will be accessible at: `http://127.0.0.1:5000/`

## 🛠️ API Endpoints
### 🔹 Segment an Image
TODO: Add this.

## 🏗️ Understanding the Structure
### **1. Data folder (`data/`)**
- Contains the data in two subfolders:
  - `meso-scale/` for images of corals
  - `micro-scale/` for CT scans (TODO: Add more details for file structure!)

### **2. Database schemas (`database/*.py`)**
- Defines **database tables** using SQLAlchemy.
- Example: `SegmentedImage` stores segmentation results.

### **3. API Schemas (`schemas/*.py`)**
- Handles **API request validation & response formatting** using Marshmallow.
- Ensures data consistency in API responses.

### **4. Routes (`routes/*.py`)**
- **Handles API requests** and calls services.
- **Does not implement functional code**.
- Uses schemas for JSON serialization.
- Example: `sam2_endpoints.py` contains routes for image segmentation.

### **5. Services (`services/*.py`)**
- Implements **business logic** for backend functionality. 
- **Does not handle API requests directly**.
- Disconnected from routes for better maintainability.
- Example: `sam2.py` contains the SAM model for image segmentation.

## Understanding the workflow
1. **API Request**: User sends a request to the API endpoint.
2. **Route**: Receives the request, validates it using *schemas*, and calls the *service*.
   1. **Schema**: Validates the request data and extracts into a usable format.
   2. **Service**: Implements the business logic, interacts with the database, does computations and returns the result.
   3. **Database**: Stores and retrieves data using SQLAlchemy models.
   4. **Service**: Returns the result to the route.
   5. **Schema**: Puts the result in valid JSON response format.
3. **Route**: Sends the response back to the user.
4. Done!
## 📜 License
This project is licensed under the AGPL3 License.

---

