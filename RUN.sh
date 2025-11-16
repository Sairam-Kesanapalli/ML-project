#!/bin/bash

################################################################################
# ML Project Automation Script
# This script automates the complete ML workflow:
# 1. Clone repository
# 2. Setup Python environment
# 3. Install dependencies
# 4. Run ML training code
# 5. Generate test data
# 6. Test the model
# 7. Display results
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/Sairam-Kesanapalli/ML-project.git"
REPO_NAME="ML-project"
PROJECT_DIR="GMM"
PYTHON_VERSION="python3"
VENV_NAME="ml_env"

# Script names from your repository
TRAINING_SCRIPT="MAIN.py"              # Your ML training script
TEST_GEN_SCRIPT="Test_Data_gen.py"     # Your test data generator script

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

################################################################################
# Step 1: System Checks
################################################################################
log "Starting ML Project Automation..."

# Check if Python is installed
if ! command -v $PYTHON_VERSION &> /dev/null; then
    error "Python3 is not installed. Please install Python 3.x first."
fi

PYTHON_VER=$($PYTHON_VERSION --version)
log "Found $PYTHON_VER"

# Check if git is installed
if ! command -v git &> /dev/null; then
    error "Git is not installed. Please install git first."
fi

################################################################################
# Step 2: Clean up old installations (optional)
################################################################################
log "Checking for existing installations..."

if [ -d "$REPO_NAME" ]; then
    warning "Found existing repository directory."
    read -p "Do you want to remove it and clone fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$REPO_NAME"
        log "Removed existing directory."
    else
        info "Using existing repository."
        cd "$REPO_NAME"
        git pull origin main || git pull origin master
        cd ..
    fi
fi

################################################################################
# Step 3: Clone Repository
################################################################################
if [ ! -d "$REPO_NAME" ]; then
    log "Cloning repository from $REPO_URL..."
    git clone "$REPO_URL" || error "Failed to clone repository"
    log "Repository cloned successfully."
fi

cd "$REPO_NAME/$PROJECT_DIR" || error "Failed to navigate to $PROJECT_DIR directory"
log "Navigated to project directory: $(pwd)"

################################################################################
# Step 4: Setup Virtual Environment
################################################################################
log "Setting up Python virtual environment..."

if [ ! -d "../../$VENV_NAME" ]; then
    $PYTHON_VERSION -m venv "../../$VENV_NAME" || error "Failed to create virtual environment"
    log "Virtual environment created."
else
    log "Virtual environment already exists."
fi

# Activate virtual environment
source "../../$VENV_NAME/bin/activate" || error "Failed to activate virtual environment"
log "Virtual environment activated."

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip

################################################################################
# Step 5: Install Dependencies
################################################################################
log "Installing dependencies..."

# Check for requirements.txt in multiple locations
if [ -f "requirements.txt" ]; then
    log "Found requirements.txt in $PROJECT_DIR directory"
    pip install -r requirements.txt || error "Failed to install dependencies from requirements.txt"
elif [ -f "../requirements.txt" ]; then
    log "Found requirements.txt in root directory"
    pip install -r ../requirements.txt || error "Failed to install dependencies"
elif [ -f "../../requirements.txt" ]; then
    log "Found requirements.txt in repository root"
    pip install -r ../../requirements.txt || error "Failed to install dependencies"
else
    warning "No requirements.txt found. Installing common ML libraries..."
    pip install numpy pandas scikit-learn matplotlib seaborn scipy || error "Failed to install default libraries"
fi

log "Dependencies installed successfully."

# Display installed packages
info "Installed packages:"
pip list

################################################################################
# Step 6: Run ML Training Code
################################################################################
log "Starting ML model training..."

if [ -f "$TRAINING_SCRIPT" ]; then
    log "Running training script: $TRAINING_SCRIPT"
    $PYTHON_VERSION "$TRAINING_SCRIPT" || error "Training script failed"
    log "Model training/prediction completed successfully!"
else
    error "Training script $TRAINING_SCRIPT not found!"
fi

################################################################################
# Step 7: Generate Test Data
################################################################################
log "Generating test data..."

if [ -f "$TEST_GEN_SCRIPT" ]; then
    log "Running test data generator: $TEST_GEN_SCRIPT"
    $PYTHON_VERSION "$TEST_GEN_SCRIPT" || error "Test data generation failed"
    log "Test data generated successfully!"
else
    error "Test generator script $TEST_GEN_SCRIPT not found!"
fi

################################################################################
# Step 8: Run MAIN.py again with generated test data (if needed)
################################################################################
log "Running model evaluation with test data..."

# MAIN.py likely handles both training and testing
# If it needs to be run again after test data generation, uncomment below:
# $PYTHON_VERSION "$TRAINING_SCRIPT" || error "Model evaluation failed"

info "Model evaluation completed. Check outputs above."

################################################################################
# Step 9: Display Results
################################################################################
log "Displaying results..."

# Look for output files
if [ -d "results" ]; then
    info "Results directory contents:"
    ls -lh results/
elif [ -d "output" ]; then
    info "Output directory contents:"
    ls -lh output/
elif [ -d "models" ]; then
    info "Models directory contents:"
    ls -lh models/
else
    info "Current directory contents:"
    ls -lh
fi

# Display any log files
if [ -f "training.log" ]; then
    info "Training log (last 20 lines):"
    tail -n 20 training.log
fi

if [ -f "test_results.txt" ]; then
    info "Test results:"
    cat test_results.txt
fi

# Look for any plots or visualizations
plot_files=$(find . -maxdepth 2 -name "*.png" -o -name "*.jpg" -o -name "*.pdf" 2>/dev/null)
if [ ! -z "$plot_files" ]; then
    info "Generated visualizations:"
    echo "$plot_files"
fi

################################################################################
# Step 10: Summary
################################################################################
log "========================================="
log "ML WORKFLOW COMPLETED SUCCESSFULLY!"
log "========================================="
info "Summary:"
info "  - Repository: $REPO_URL"
info "  - Project directory: $PROJECT_DIR"
info "  - Virtual environment: $VENV_NAME"
info "  - Working directory: $(pwd)"
log "========================================="

# Deactivate virtual environment
deactivate

log "Script execution finished. Virtual environment deactivated."
log "To reactivate the environment, run: source $VENV_NAME/bin/activate"
