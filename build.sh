# Start the build process
echo '==> Starting build..'

# # Setup the virtual environment 
# echo '-- Setting up environment..'
# python3 -m venv my_env
# source my_env/bin/activate

# Create Data Directories
echo '-- Creating data directories..'
rm -Rf data # Remove the data file if it already exists
mkdir -p data data/raw data/processed # Create new directory for processed data from raw data 

# Import raw data
echo '-- Importing raw data..'
yes | python3 src/data/import_raw_data.py # Automatically answer yes to do you want to create a new file ?

echo '-- Processing raw data..'
python3 src/data/make_dataset.py data/raw data/processed

# Build features
echo '-- Building features..'
python3 src/features/build_features.py

# Train model
echo '-- Training model..'
python3 src/models/train_model.py

echo '==> Finished build!'
