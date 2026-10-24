echo "Upgrading pip..."
pip install --upgrade pip --break-system-packages

echo "Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages
pip install -e . --break-system-packages