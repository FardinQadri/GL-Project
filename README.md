# GL_Model Deployment

## Introduction
This repository contains a solution for training and deploying a classification model on the Employee attrition rate dataset. The model is wrapped in a Flask server, containerized using Docker, and shared via Docker Hub. This setup allows easy access and utilization of the model across the organization.

## Requirements
- Python 
- Docker
- Flask
- Git Hub

## Checklist
-  Choose and train a classification model on the Employee dataset.
-  Implement saving and loading functionalities for the trained model.
-  Document Python functions for team members to use the model.
-  Document Flask API usage for organization-wide accessibility.
-  Adhere to organization coding guidelines.
-  Utilize GitHub for code version control.
-  Implement CI/CD pipelines for automated testing.
-  Push Docker images to Docker Hub according to company guidelines.

## Project Structure
- `model.py`: Contains code for training and saving the model.
- `app.py`: Loads the model and sets up a Flask server with documentation.
- `.github/workflows/`: main.yml

## Usage
### Python Function Usage
To use the model within Python scripts:
1. Load the model using `load_model()` function.
2. Call the `predict()` function with the required input.

### Flask API Usage
To access the model via Flask API:
1. Run the Flask server using `python load_model_flask.py`.
2. Access the API endpoint with appropriate data and receive predictions.

## Getting Started
1. Clone this repository
2. Install necessary dependencies: `pip install -r requirements.txt`
3. Train the model using `model.py`.
4. Set up the Flask server using `app.py`.
5. Test functionality locally.
6. Push changes to GitHub repository.

## CI/CD Pipelines
The repository includes GitHub Actions workflows for automated testing and Docker image creation and push. Upon pushing changes to the repository, the workflows will automatically run and ensure code functionality and Docker image availability on Docker Hub.
