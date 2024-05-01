echo '==> Starting build..'

echo '-- Installing requirements..'
pip install -r requirements.txt

echo '-- Setting up environment..'
python3 -m venv my_env
source my_env/bin/activate

echo '-- Creating data directories..'
rm -Rf data
mkdir -p data data/raw data/processed

echo '-- Importing raw data..'
yes | python3 src/data/import_raw_data.py

echo '-- Processing raw data..'
python3 src/data/make_dataset.py data/raw data/processed

echo '-- Building features..'
python3 src/features/build_features.py

echo '-- Training model..'
python3 src/models/train_model.py

# echo '-- Testing model..'
# python3 src/models/predict_model.py

echo '==> Finished build!'