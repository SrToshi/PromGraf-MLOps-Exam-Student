N ?= 30

all: 
	docker-compose up --build -d

stop: 
	docker-compose down

evaluation:
	docker-compose up -d --build evaluation

traffic:
	chmod +x generate_predict_traffic.sh
	./generate_predict_traffic.sh $(N)

fire-alert:
	chmod +x fire_alert.sh
	./fire_alert.sh

help:
	@echo "Available targets:"
	@echo "  make all        - Start all services (API, Prometheus, Grafana, Node Exporter)"
	@echo "  make stop       - Stop all services"
	@echo "  make evaluation - Run run_evaluation.py to update model metrics in Prometheus"
	@echo "  make traffic    - Send N simulated requests to /predict (default N=30)"
	@echo "                    Usage: make traffic N=50"
	@echo "  make fire-alert - Trigger the 'HighModelRMSE' Grafana alert with synthetic data"
	@echo "                    (forces model_rmse_score > 20; run 'make evaluation' after to restore normal metrics)"
