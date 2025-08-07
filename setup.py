"""
Project setup script - Creates initial project structure and files
"""
import os
import sys
from pathlib import Path
import argparse
import subprocess

def create_project_structure(project_name="forex-trading-ml-pipeline"):
    """Create complete project structure"""
    
    structure = {
        "": [
            "README.md",
            "requirements.txt", 
            "setup.py",
            ".gitignore",
            "Dockerfile",
            "docker-compose.yml",
            "LICENSE"
        ],
        "config": ["config.yaml"],
        "src": ["__init__.py"],
        "src/data": ["__init__.py", "data_collector.py", "preprocessor.py"],
        "src/models": ["__init__.py", "black_scholes.py", "bayesian_models.py", "ml_models.py"],
        "src/utils": ["__init__.py", "helpers.py"],
        "src/api": ["__init__.py", "app.py"],
        "src/dashboard": ["__init__.py", "app.py"],
        "scripts": ["train_model.py", "predict.py", "deploy.py", "setup_project.py"],
        "tests": ["__init__.py", "test_data.py", "test_models.py", "test_api.py"],
        "notebooks": [
            "01_data_exploration.ipynb",
            "02_model_development.ipynb", 
            "03_backtesting.ipynb"
        ],
        "data": [],
        "data/raw": [".gitkeep"],
        "data/processed": [".gitkeep"],
        "models": [".gitkeep"],
        "logs": [".gitkeep"],
        ".github/workflows": ["ci-cd.yml"]
    }
    
    print(f"Creating project structure for: {project_name}")
    
    # Create base directory
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    
    os.chdir(project_name)
    
    # Create directory structure
    for directory, files in structure.items():
        if directory:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        
        for file in files:
            file_path = os.path.join(directory, file) if directory else file
            
            # Create file if it doesn't exist
            if not os.path.exists(file_path):
                Path(file_path).touch()
                print(f"✓ Created file: {file_path}")
    
    print(f"\n✅ Project structure created successfully!")
    print(f"📁 Project location: {os.getcwd()}")
    print(f"\nNext steps:")
    print(f"1. cd {project_name}")
    print(f"2. pip install -r requirements.txt")
    print(f"3. python scripts/train_model.py")
    print(f"4. uvicorn src.api.app:app --reload")

def setup_virtual_environment():
    """Setup virtual environment"""
    print("Setting up virtual environment...")
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
        
        # Activation instructions
        if sys.platform.startswith('win'):
            activate_cmd = "venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
            
        print(f"To activate: {activate_cmd}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")

def install_dependencies():
    """Install project dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")

def main():
    parser = argparse.ArgumentParser(description='Setup Forex Trading ML Project')
    parser.add_argument('--name', type=str, default='forex-trading-ml-pipeline', 
                       help='Project name')
    parser.add_argument('--venv', action='store_true', 
                       help='Create virtual environment')
    parser.add_argument('--install', action='store_true', 
                       help='Install dependencies')
    
    args = parser.parse_args()
    
    print("🚀 Forex Trading ML Pipeline Setup")
    print("=" * 40)
    
    # Create project structure
    create_project_structure(args.name)
    
    # Setup virtual environment
    if args.venv:
        setup_virtual_environment()
    
    # Install dependencies  
    if args.install:
        install_dependencies()
    
    print("\n🎉 Setup completed!")
    print("Check README.md for detailed usage instructions.")

if __name__ == "__main__":
    main()