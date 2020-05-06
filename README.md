To download data open bash terminal in project dir and run 
python ./app/training/download_script.py

To train model on data open bash terminal in project dir and run
python ./app/training/train_script.py

*Warning:* Training take an incredibly long time depending on your resources
feel free to modify the train script for number of desired training epochs,
model parameters, vocab size, batch size, etc.

Refer to model_trainer for available methods -- checkpoints after every epoch 
will be uploaded to app/training/models/model_checkpoints/ folder.

*Note:* Before starting app it is recommended you only have the initial checkpoint
file and the desired checkpoint number in model_checkpoints before building container.

*ex:* checkpoint ckpt-35.data-0000-of-00002 ckpt-35.data-00001-of-00002 ckpt-35.index

To start app in docker run start.sh in bash terminal.
*Warning:* May take awhile to build docker image first.

To stop app run docker ps then docker stop <pid> of the app.

To start app without docker open a bash terminal in project dir and do following:
    # sets up env
    - cd app
    - python3 -m venv env
    - source ./env/b*/a*
    - pip install --upgrade pip=19.3
    - pip install -r requirements.txt
    - pip install git+https://github.com/Maluuba/nlg-eval.git@master
    - nlg-eval --setup

    #runs app
    python main.py

Go to http://127.0.0.1:5001/ to see web app,
or http://127.0.0.1:5001/api to see the the swagger docs for the api.
    
    Query with :
    curl -v -X GET -F "file=@<path/to/image>"  http://127.0.0.1:5001/api/caption

    ex: curl -v -X GET -F "file=@./zebras.jpeg"  http://127.0.0.1:5001/api/caption

    or

    curl -v -X GET -F "file=@./zebras.jpeg"  http://127.0.0.1:5001/api/caption?candidates=5

    Where candidates is the number of captions you want for one image, up to 10.

To run tests open bash terminal in project dir and run pytest ./app/training/models/test_models.py

*Note:* Some tests take a long time, put @pytest.mark.skip on any test you want to skip.
*Prereqs:* Some tests require you to have run the download_script.py in app/training first.
