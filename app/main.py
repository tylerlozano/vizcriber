import logging
import os
import uuid

import werkzeug
from werkzeug.exceptions import BadRequest
from flask import (Blueprint, Flask, flash, redirect, render_template, request,
                   url_for)
from flask_bootstrap import Bootstrap
from flask_restplus import Api, Resource, fields, reqparse

from training.models import model_driver, model_trainer

app = Flask(__name__)
Bootstrap(app)

# set basic logging level
logging.basicConfig(level=logging.DEBUG)

# disable Try it Out in swagger dors
app.config.SWAGGER_SUPPORTED_SUBMIT_METHODS = []
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'i challenge you to a slap bass battle'

# display swagger doc at url_prefix
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(blueprint, version='1.0', title='Vizcriber API',
          description='An image captioning API')
app.register_blueprint(blueprint)

# namespace
ns = api.namespace('api', description='caption images')

model_id = 'ckpt-35'
# instantiate model
app.logger.info("Instantiating model")
mt = model_trainer.ModelTrainer(training=False)
# indicate what version model to load
mt.load_model(model_id)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
"""
Routes
"""

# todo: add file verification
@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        if 'file' not in request.files:
            app.logger.debug("File not in request files")
            flash(' No file part.')
            return redirect(url_for('index'))

        file = request.files['file']

        if file.filename == '':
            app.logger.debug("File not selected")
            flash(' No selected file.')
            return redirect(url_for('index'))

        if str(file.filename).split('.')[-1] not in ALLOWED_EXTENSIONS:
            app.logger.debug("File has invalid extension")
            flash(f" Invalid filetype. Try file with "
                  f".{' .'.join(ALLOWED_EXTENSIONS)} extension instead.")
            return redirect(url_for('index'))

        if file:
            app.logger.info(f"Loading image file {file.filename}")
            image_name = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join('static', image_name)
            file.save(image_path)
            app.logger.info("Running image through model")
            caption = model_driver.get_prediction(image_path, mt.ckpt)
            result = {
                'caption': caption[0],
                'image_path': image_path
            }
            app.logger.info("Serving result")
            return render_template('show.html', result=result)
    return render_template('index.html')


"""
API ENDPOINTS
"""


class FileField(fields.Raw):
    __schema_type__ = 'image'


input_fields = api.model('caption_inputs', {
    'file': FileField(required=True, description="An image to be captioned."),
    'candidates': fields.Integer(min=0, max=10,
                                 description="Number of captions to output.")
})

output_fields = api.model('caption_outputs', {
    'captions': fields.List(fields.String(max_length=20,
                                          description="A list of different captions for specified image."))
})


# ex : curl -v -X GET -F "file=@./zebras.jpeg" http://127.0.0.1:5000/v1/caption
# ex : curl -v -X GET -F "file=@./zebras.jpeg"  http://127.0.0.1:5000/api/caption?candidates=5
@api.route('/caption')
class Caption(Resource):
    @api.marshal_with(output_fields)
    @api.expect(input_fields)
    def get(self):
        parse = reqparse.RequestParser(bundle_errors=True)
        parse.add_argument('file',
                           type=werkzeug.datastructures.FileStorage,
                           required=True,
                           location='files',
                           help="File must be image.")
        parse.add_argument('candidates',
                           type=int,
                           default=0,
                           choices=list(range(11)),
                           help='Number of candidate captions ranges from 0 to 10.')

        args = parse.parse_args()

        app.logger.info((f"Processing /caption request with "
                         f"{args['candidates']} candidates"))

        file = args['file']

        if file.filename == '':
            app.logger.debug("File not selected in /caption request")
        elif not str(file.filename).split('.')[-1] in ALLOWED_EXTENSIONS:
            app.logger.debug(f"File with invalid extension"
                             f"{str(file.filename).split('.')[-1]} in /caption request")
            e = BadRequest('Input payload validation failed.')
            e.data = {'errors': {'file': f"File has invalid extension ."
                                 f"{str(file.filename).split('.')[-1]}", }, }
            raise e
        else:
            image_name = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join('static', image_name)
            file.save(image_path)

            captions = model_driver.get_prediction(image_path,
                                                   mt.ckpt,
                                                   args['candidates'])

            result = {
                'captions': captions
            }

            return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
