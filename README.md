*Warning:* Before starting app it is recommended you only have the initial checkpoint
file and the desired checkpoint number in model_checkpoints before building container.

*ex:* checkpoint ckpt-35.data-0000-of-00002 ckpt-35.data-00001-of-00002 ckpt-35.index

<h3> start app to serve model -- production mode </h3>
# open a bash terminal in project directory and do following:
./start.sh

<h3> stop app </h3>
# open a bash terminal in project directory and do following:
docker ps
# copy <pid> of container
docker stop <pid>


<h3> Set-up virtualenv to train model</h3>
   # open a bash terminal in project directory and do following:
   * cd app
   * python3 -m venv env
   * source ./env/b*/a*
   * pip install --upgrade pip=19.3
   * pip install -r requirements.txt
   * pip install git+https://github.com/Maluuba/nlg-eval.git@master
   * nlg-eval --setup

# to download data 
python ./training/download_script.py

# to train -- modify to fit custom specifications
python ./training/train_script.py

#  to run app 
python main.py

# to test model
Go to http://127.0.0.1:5001/ to see web app,
or http://127.0.0.1:5001/api to see the the swagger docs for the api.
    
    Query with :
    curl -v -X GET -F "file=@<path/to/image>"  http://127.0.0.1:5001/api/caption

    ex: curl -v -X GET -F "file=@./zebras.jpeg"  http://127.0.0.1:5001/api/caption

    or

    curl -v -X GET -F "file=@<path/to/image>"  http://127.0.0.1:5001/api/caption?candidates=5

    Where candidates is the number of captions you want for one image, up to 10.

# To run unit tests
pytest training/models/test_models.py

*Note:* Some tests take a long time, put @pytest.mark.skip on any test you want to skip.
*Prereqs:* Some tests require you to have run the download_script.py in app/training first.
