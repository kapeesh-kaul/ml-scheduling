run_scheduler:
	python scheduler/run_scheduler.py $(ARGS)
train_model:
	python scheduler/models_utils.py
predict:
	python app.py
