#!/bin/bash
# Training script helper
# Usage: ./train.sh

echo "🚀 Starting Deepfake Detection Model Training..."
echo ""

cd "$(dirname "$0")/pytorch"

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found!"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found!"
    exit 1
fi

echo "📋 Configuration:"
cat config.yaml | grep -E "epochs|learning_rate|batch_size" | sed 's/^/   /'
echo ""

# Run training
echo "🏋️  Starting training..."
python train_improved.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📦 Model saved to: ../../models/xception_deepfake.pth"
    echo ""
    echo "💡 You can now:"
    echo "   1. Copy the model to other devices"
    echo "   2. Start the backend server: cd .. && python app.py"
else
    echo ""
    echo "❌ Training failed. Check the logs above for errors."
    exit 1
fi
