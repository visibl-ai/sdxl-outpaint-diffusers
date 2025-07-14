#!/bin/bash

# Modal deployment script
# Usage: ./modal_deploy.sh [dev|prod] [web|inference|both]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [dev|prod] [web|inference|both]"
    echo ""
    echo "Environment Options:"
    echo "  dev   - Deploy to development environment using env.dev"
    echo "  prod  - Deploy to production environment using env.prod"
    echo ""
    echo "App Options:"
    echo "  web       - Deploy only the web app (modal_web.py)"
    echo "  inference - Deploy only the inference app (modal_inference.py)"
    echo "  both      - Deploy both web and inference apps (default)"
    echo ""
    echo "Examples:"
    echo "  $0 dev web        # Deploy web app to development"
    echo "  $0 prod inference # Deploy inference app to production"
    echo "  $0 dev both       # Deploy both apps to development"
    echo "  $0 prod           # Deploy both apps to production (default)"
}

# Check if environment argument is provided
if [ $# -eq 0 ]; then
    print_error "No environment specified"
    show_usage
    exit 1
fi

ENVIRONMENT=$1
APP_TYPE=${2:-both}  # Default to 'both' if not specified

# Validate environment argument
if [ "$ENVIRONMENT" != "dev" ] && [ "$ENVIRONMENT" != "prod" ]; then
    print_error "Invalid environment: $ENVIRONMENT"
    show_usage
    exit 1
fi

# Validate app type argument
if [ "$APP_TYPE" != "web" ] && [ "$APP_TYPE" != "inference" ] && [ "$APP_TYPE" != "both" ]; then
    print_error "Invalid app type: $APP_TYPE"
    show_usage
    exit 1
fi

# Set environment file based on deployment target
if [ "$ENVIRONMENT" = "dev" ]; then
    ENV_FILE=".env.dev"
    DEPLOY_NAME_PREFIX="sdxl-outpaint-dev"
else
    ENV_FILE=".env.prod"
    DEPLOY_NAME_PREFIX="sdxl-outpaint-prod"
fi

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    print_error "Environment file $ENV_FILE not found"
    exit 1
fi

print_status "Deploying to $ENVIRONMENT environment..."
print_status "Using environment file: $ENV_FILE"
print_status "App type: $APP_TYPE"

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    print_error "Modal CLI is not installed. Please install it first."
    exit 1
fi

# Check if we're logged into modal
if ! modal profile current &> /dev/null; then
    print_error "Not logged into Modal. Please run 'modal token new' first."
    exit 1
fi

# Deploy using modal
print_status "Starting deployment..."

# Source the environment file
set -a  # automatically export all variables
source "$ENV_FILE"
set +a

# Function to deploy an app
deploy_app() {
    local app_file=$1
    local app_name=$2

    print_status "Deploying $app_name..."
    if modal deploy "$app_file"; then
        print_success "Successfully deployed $app_name!"
        print_status "You can view your app in the Modal dashboard"
    else
        print_error "Failed to deploy $app_name!"
        return 1
    fi
}

# Deploy based on app type
case "$APP_TYPE" in
    "web")
        deploy_app "modal_web.py" "${DEPLOY_NAME_PREFIX}-web"
        ;;
    "inference")
        deploy_app "modal_inference.py" "${DEPLOY_NAME_PREFIX}-inference"
        ;;
    "both")
        print_status "Deploying both web and inference apps..."
        deploy_app "modal_web.py" "${DEPLOY_NAME_PREFIX}-web"
        deploy_app "modal_inference.py" "${DEPLOY_NAME_PREFIX}-inference"
        print_success "Successfully deployed both apps to $ENVIRONMENT environment!"
        ;;
esac
