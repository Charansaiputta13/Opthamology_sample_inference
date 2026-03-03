#!/bin/bash
# Quick deployment script for DR Classification System
# Usage: ./deploy.sh [docker|heroku|gcloud|azure]

set -e

DEPLOY_TYPE=${1:-docker}

echo "🚀 DR Classification Deployment Script"
echo "======================================"
echo "Deployment Type: $DEPLOY_TYPE"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

case $DEPLOY_TYPE in
  docker)
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t dr-classifier:latest .
    
    echo -e "${YELLOW}Running Docker container...${NC}"
    docker run -p 8501:8501 \
      -v $(pwd)/data:/app/data \
      -v $(pwd)/outputs:/app/outputs \
      dr-classifier:latest
    
    echo -e "${GREEN}✅ Running at http://localhost:8501${NC}"
    ;;
    
  docker-compose)
    echo -e "${YELLOW}Starting Docker Compose...${NC}"
    docker-compose up -d
    
    echo -e "${GREEN}✅ Running at http://localhost:8501${NC}"
    echo -e "${YELLOW}View logs: docker-compose logs -f${NC}"
    echo -e "${YELLOW}Stop: docker-compose down${NC}"
    ;;
    
  heroku)
    echo -e "${YELLOW}Checking Heroku CLI...${NC}"
    if ! command -v heroku &> /dev/null; then
      echo -e "${RED}❌ Heroku CLI not installed${NC}"
      echo "Install from: https://devcenter.heroku.com/articles/heroku-cli"
      exit 1
    fi
    
    echo -e "${YELLOW}Logging in to Heroku...${NC}"
    heroku login
    
    echo -e "${YELLOW}Creating Heroku app (if needed)...${NC}"
    heroku apps:create dr-classifier || true
    
    echo -e "${YELLOW}Setting Docker buildpack...${NC}"
    heroku buildpacks:set heroku/docker
    
    echo -e "${YELLOW}Deploying to Heroku...${NC}"
    git push heroku master
    
    echo -e "${GREEN}✅ Deployed to Heroku${NC}"
    heroku open
    ;;
    
  gcloud)
    echo -e "${YELLOW}Checking Google Cloud SDK...${NC}"
    if ! command -v gcloud &> /dev/null; then
      echo -e "${RED}❌ Google Cloud SDK not installed${NC}"
      echo "Install from: https://cloud.google.com/sdk/docs/install"
      exit 1
    fi
    
    PROJECT_ID=$(gcloud config get-value project)
    echo -e "${YELLOW}Building for Google Cloud Run...${NC}"
    gcloud builds submit --tag gcr.io/$PROJECT_ID/dr-classifier
    
    echo -e "${YELLOW}Deploying to Google Cloud Run...${NC}"
    gcloud run deploy dr-classifier \
      --image gcr.io/$PROJECT_ID/dr-classifier \
      --platform managed \
      --region us-central1 \
      --port 8501 \
      --memory 2Gi \
      --cpu 2
    
    echo -e "${GREEN}✅ Deployed to Google Cloud Run${NC}"
    gcloud run services describe dr-classifier --region us-central1
    ;;
    
  azure)
    echo -e "${YELLOW}Checking Azure CLI...${NC}"
    if ! command -v az &> /dev/null; then
      echo -e "${RED}❌ Azure CLI not installed${NC}"
      echo "Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
      exit 1
    fi
    
    RESOURCE_GROUP="dr-classifier-rg"
    REGISTRY_NAME="drclassifieracr"
    
    echo -e "${YELLOW}Creating resource group...${NC}"
    az group create --name $RESOURCE_GROUP --location eastus || true
    
    echo -e "${YELLOW}Creating Azure Container Registry...${NC}"
    az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic || true
    
    echo -e "${YELLOW}Building and pushing image...${NC}"
    az acr build --registry $REGISTRY_NAME --image dr-classifier:latest .
    
    echo -e "${YELLOW}Deploying to Azure Container Instances...${NC}"
    az container create \
      --resource-group $RESOURCE_GROUP \
      --name dr-classifier \
      --image ${REGISTRY_NAME}.azurecr.io/dr-classifier:latest \
      --ports 8501 \
      --environment-variables PORT=8501
    
    echo -e "${GREEN}✅ Deployed to Azure${NC}"
    az container show --resource-group $RESOURCE_GROUP --name dr-classifier --query ipAddress.fqdn
    ;;
    
  *)
    echo -e "${RED}❌ Unknown deployment type: $DEPLOY_TYPE${NC}"
    echo ""
    echo "Usage: ./deploy.sh [docker|docker-compose|heroku|gcloud|azure]"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh docker              # Local Docker"
    echo "  ./deploy.sh docker-compose      # Docker Compose"
    echo "  ./deploy.sh heroku              # Deploy to Heroku"
    echo "  ./deploy.sh gcloud              # Deploy to Google Cloud Run"
    echo "  ./deploy.sh azure               # Deploy to Azure"
    exit 1
    ;;
esac
