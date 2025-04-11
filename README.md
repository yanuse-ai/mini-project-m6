# mini-project-m6
Mini Project 2: Continuous Integration for Model Training
employee_attrition_model

Objective: Implement a CI/CD pipeline to continuously train and deploy an ML model.
1. Create a Git repository for the project and include the ML model code, training
data, and evaluation scripts.
2. Implement a GitHub Actions workflow to trigger model training whenever new
training data is pushed to the repository.
3. Use Docker containers to encapsulate the training environment and
dependencies.
4. After successful training, run evaluation scripts to assess model performance.
5. Build the Docker image for the model training environment and push it to Docker
Hub.
6. Use GitHub action push/merge to trigger model deployment on AWS EC2
whenever a new Docker image is pushed to Docker Hub.
