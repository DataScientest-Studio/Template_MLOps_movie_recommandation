# Start the test build process
echo '==> Starting tests..'

# Run the prediction test
echo '-- Running Prediction test ..'
python3 src/models/predict_model.py

# Run unit tests
echo '-- Running unit tests..'
pytest

# Finish the test build process
echo '==> Finished tests!'
