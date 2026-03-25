# FastAPI Backend for Coral Analysis

## 📌 Overview
This repository contains a FastAPI backend for image analysis. It follows a modular structure with clear separation between **routes, services, models, and schemas** to ensure scalability and maintainability.

---
## 🏗️ Project Structure
```
app/
│── database         # Database models (SQLAlchemy). Define the database scheme
│   |── coral.py     
│── schemas          # API request/response validation (PyDantic). Define the data format for requests/responses
│   │── *.py         
│── routes/          # API endpoints (FastAPI Routers)
│   │── *.py         
│── services/        # Defines actual functionalities. Disconnected from routes.
│   │── *.py  
│── __init__.py      # FastAPI app factory function. Builds the app and initialises the database
data/                # Data folder. Saves images and CT Scans, and all other kind of data
|── Dataset folders  # This is where dataset folders appear after upload
│   │── images       # Folder containing the uploaded images in their original file format
│   │── masks        # Folder containing finalized masks as .png files
|── thumbnails       # Low resolution copy of the images for faster loading speeds      
|── database.db      # SQLite database
tests/*              # Tests for services and routes
config.py            # Configuration settings (database, environment variables, paths etc.)
requirements.txt     # Project dependencies
Readme.md            # This file
main.py              # Entry point to start the Fastapi application
```

---
## 🛠️ Installation & Setup
### Option 1: Using Docker Compose
#### 1. Build and Run the Project (Production Mode)
Ensure you have Docker and Docker Compose installed on your system. Then, use the following command to build and run the project:
```sh
docker-compose up --build
```
This command will build the Docker image as specified in the `Dockerfile` and start the service defined in the `docker-compose.yml` file.

#### 2. Development Mode with Hot Reload
For development with automatic code reloading (no need to rebuild on code changes), use the development compose file:
```sh
docker-compose -f docker-compose.dev.yml up --build
```
This will:
- Mount your source code as volumes, so changes are reflected immediately
- Enable uvicorn's `--reload` flag for automatic server restart on code changes
- Keep all your data, logs, and weights persisted between restarts

**Note:** After the initial build, you can use `docker-compose -f docker-compose.dev.yml up` (without `--build`) for faster startup. The container will automatically restart when you modify any Python files in the `app/` directory or other source files.

### Option 2: Using Podman
Install podman and build the image using
```sh
podman build -t coral-backend .
```
Then run the container using
```sh
podman run -p 8000:8000 coral-backend
```
### Option 3: Using venv
#### 1. Clone the Repository
```sh
git clone <repo-url>
cd <repo-folder>
```

#### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

#### 3. Install Dependencies
> **Note:** If you want to use CUDA for GPU acceleration, make sure to install the correct version of PyTorch that 
> matches your CUDA version **before** installing the requirements. You can find the correct command for your system 
> [here](https://pytorch.org/get-started/locally/). The requirements.txt file contains the CPU version of PyTorch, which
> will be installed by default, if you do not install the CUDA version first.

Install the required packages:
```sh
pip install -r requirements.txt
```

#### 4. Run the Flask Application
```shell
fastapi run main.py
```
or for developer mode
```shell
fastapi dev main.py
```
>Server will be accessible at: http://127.0.0.1:8000
>
>Swagger Docs can be accessed at: http://127.0.0.1:8000/docs
---
## 🏗️ Understanding the Structure
### **1. Data folder (`data/`)**
- Contains subfolders for each dataset
- Contains the database file

### **2. Database schemas (`database/*.py`)**
- Defines **database tables** using SQLAlchemy.

### **3. API Schemas (`schemas/*.py`)**
- Handles **API request validation & response formatting** using PyDantic.
- Ensures data consistency in API responses.

### **4. Routes (`routes/*.py`)**
- **Handles API requests** and calls services.
- **Does not implement functional code**.
- Uses schemas for JSON serialization.

### **5. Services (`services/*.py`)**
- Implements data processing tasks for backend functionality. 
- **Does not handle API requests directly**.
- Disconnected from routes for better maintainability.

---
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
---
## Endpoint Docs
For an endpoint documentation run the API and visit the dynamic SwaggerUI docs at `http://{your_api_url}/docs`.

---
## 📜 License
This project is licensed under the AGPL3 License.

---

