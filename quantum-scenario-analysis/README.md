# Quantum Scenario Analysis

This project integrates a Quantum Risk Aggregation backend with a React frontend.

## Project Structure

```
├── backend/                # Python Flask Backend
│   ├── quantum_risk_backend.py    # Main quantum analysis code
│   ├── flask_api_server.py        # API server
│   └── requirements.txt           # Python dependencies
│
├── frontend/               # React Frontend
│   ├── src/                # Frontend source code
│   ├── public/
│   ├── package.json
│   └── ...                 # Vite configuration files
```

## Setup Instructions

### 1. Backend Setup

Prerequisites: Python 3.12+

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
   *(Note: If `backend` directory does not exist, look for `backend_new` and rename it to `backend` after removing the old `Backend` folder)*.

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API server:
   ```bash
   python flask_api_server.py
   ```
   The API will be available at `http://localhost:5000`.

### 2. Frontend Setup

Prerequisites: Node.js & npm

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The app will typically run on `http://localhost:8080` (check terminal output).

## Troubleshooting

- **File Locks**: If you encounter issues (e.g., "Access Denied") during setup, unsure all previous `node` and `python` processes are terminated.
- **Legacy Folders**: You may see `Backend` (legacy) or `src` (root) folders. You can safely delete them after verifying `backend` and `frontend` contain your files.
