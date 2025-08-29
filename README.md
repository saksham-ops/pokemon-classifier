Pokemon Classifier API
Description
This API predicts whether a given Pokémon is legendary or not, based on its stats (on Tableau) and features. It uses a machine learning model trained on various attributes such as attack, defense, special attack, special defense, catch rate, and more.

Key Features:

Accurately classifies Pokémon into legendary or non-legendary categories.
Considers a comprehensive set of features for prediction.
Offers flexibility in deployment, either on AWS App Runner or locally.

Installation
Option 1: Deployment on AWS App Runner
Create an AWS App Runner service using the provided Dockerfile.
Push the Docker image to AWS ECR (Elastic Container Registry).
Configure the App Runner service to use the ECR image.
Option 2: Running Locally
Install dependencies:
Bash
pip install -r requirements.txt
Use code with caution.
Run the API:
Bash
python app.py
Use code with caution.

Usage
Send a POST request to the API endpoint with the Pokémon's features in the request body.
The API will respond with a JSON object indicating whether the Pokémon is predicted to be legendary or not.
Example Usage
Bash
curl -X POST -H "Content-Type: application/json" -d '{"attack": 120, "defense": 95, "special_attack": 154, "special_defense": 90, "catch_rate": 3}' http://localhost:5000/predict
Use code with caution.

Or you may upload a file for Batch Prediction using route '/predict_file" and your file with predicted outputs will be downloaded in your system when request method is "POST"

DashBoard link
https://public.tableau.com/views/PokemonDashboard_17045321287470/Dashboard1?:language=en-US&:display_count=n&:origin=viz_share_link

Contact
https://www.linkedin.com/in/aryan-gaur-b49550258/)https://www.linkedin.com/in/aryan-gaur-b49550258/
"# pokemon-classifier" 
"# pokemon-classifier" 
