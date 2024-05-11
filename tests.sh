# Start the test sbuild process
echo '==> Starting build..'

echo '-- Running Prediction test ..'
python3 src/models/predict_model.py

echo '-- Running unit tests..'
pytest

echo '==> Finished build!'
