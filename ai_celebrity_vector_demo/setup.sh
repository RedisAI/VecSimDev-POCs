#!/bin/bash
# Celebrity Face Finder - Complete Setup Script
# Run this once before your lecture to set up everything!

set -e  # Exit on error

echo "=========================================="
echo "  Celebrity Face Finder - Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8+ and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 is installed${NC}"

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${YELLOW}⚠️  Kaggle API credentials not found!${NC}"
    echo ""
    echo "To set up Kaggle API:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Scroll to 'API' section and click 'Create New Token'"
    echo "3. Move kaggle.json to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Do you want to continue without downloading the dataset? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    SKIP_DOWNLOAD=true
else
    echo -e "${GREEN}✅ Kaggle credentials found${NC}"
    SKIP_DOWNLOAD=false
fi

echo ""
echo -e "${BLUE}🚀 Starting setup...${NC}"
echo ""

# Step 1: Start Redis
echo -e "${BLUE}📦 Step 1/5: Starting Redis Stack...${NC}"
docker-compose up -d
sleep 3
echo -e "${GREEN}✅ Redis Stack is running${NC}"
echo "   - Redis: localhost:6379"
echo "   - RedisInsight: http://localhost:8001"
echo ""

# Step 2: Install Python dependencies
echo -e "${BLUE}📚 Step 2/5: Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 3: Download dataset
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}📥 Step 3/5: Downloading celebrity dataset...${NC}"
    echo "   This may take a few minutes..."
    python3 1_download_dataset.py
    echo -e "${GREEN}✅ Dataset downloaded${NC}"
    echo ""
else
    echo -e "${YELLOW}⏭️  Step 3/5: Skipping dataset download${NC}"
    echo ""
fi

# Step 4: Generate embeddings
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}🧬 Step 4/5: Generating face embeddings...${NC}"
    echo "   This will take 10-30 minutes depending on your CPU..."
    echo "   ☕ Perfect time for a coffee break!"
    python3 2_generate_embeddings.py
    echo -e "${GREEN}✅ Embeddings generated${NC}"
    echo ""
else
    echo -e "${YELLOW}⏭️  Step 4/5: Skipping embedding generation${NC}"
    echo ""
fi

# Step 5: Load to Redis
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${BLUE}💾 Step 5/5: Loading embeddings into Redis...${NC}"
    python3 3_load_to_redis.py
    echo -e "${GREEN}✅ Data loaded into Redis${NC}"
    echo ""
else
    echo -e "${YELLOW}⏭️  Step 5/5: Skipping Redis loading${NC}"
    echo ""
fi

# Done!
echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "🎉 Your Celebrity Face Finder demo is ready!"
echo ""
echo "To run the demo:"
echo -e "  ${BLUE}streamlit run streamlit_app.py${NC}"
echo ""
echo "To stop Redis:"
echo -e "  ${BLUE}docker-compose down${NC}"
echo ""
echo "Next time (data is persisted):"
echo -e "  ${BLUE}docker-compose up -d && streamlit run streamlit_app.py${NC}"
echo ""
echo "=========================================="

