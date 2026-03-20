pipeline {
    agent any

    environment {
        PROJECT_ID      = "settletax"
        REGION          = "us-central1"
        SERVICE_NAME    = "settletax-classifier"
        IMAGE           = "us-central1-docker.pkg.dev/${PROJECT_ID}/${SERVICE_NAME}/classifier"
        GOOGLE_CREDENTIALS = credentials('gcp-service-account-key')
    }

    triggers {
        // Triggers on every push — GitHub webhook handles branch filtering
        githubPush()
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Authenticate GCP') {
            steps {
                sh '''
                    echo "$GOOGLE_CREDENTIALS" > /tmp/gcp-key.json
                    gcloud auth activate-service-account --key-file=/tmp/gcp-key.json
                    gcloud config set project $PROJECT_ID
                    gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE}:${BUILD_NUMBER} -t ${IMAGE}:latest ."
            }
        }

        stage('Push to Artifact Registry') {
            steps {
                sh '''
                    docker push ${IMAGE}:${BUILD_NUMBER}
                    docker push ${IMAGE}:latest
                '''
            }
        }

        stage('Deploy to Cloud Run') {
            steps {
                sh '''
                    gcloud run deploy $SERVICE_NAME \
                        --image=${IMAGE}:${BUILD_NUMBER} \
                        --region=$REGION \
                        --platform=managed \
                        --quiet
                '''
            }
        }

        stage('Cleanup') {
            steps {
                sh '''
                    rm -f /tmp/gcp-key.json
                    docker rmi ${IMAGE}:${BUILD_NUMBER} || true
                '''
            }
        }
    }

    post {
        success {
            echo "Deployed ${SERVICE_NAME}:${BUILD_NUMBER} to Cloud Run successfully."
        }
        failure {
            echo "Deployment failed. Check logs above."
        }
    }
}
