import os
import uuid

import werkzeug
from flask import Blueprint, Flask, redirect, render_template, request, url_for
from flask_bootstrap import Bootstrap
from flask_restplus import Api, Resource, fields, reqparse

from training.models import model_driver
from training.models import model_trainer

app = Flask(__name__)
Bootstrap(app)

# disable Try it Out
app.config.SWAGGER_SUPPORTED_SUBMIT_METHODS = []

# display swagger doc at url_prefix
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(blueprint, version='1.0', title='Vizcriber API',
          description='An image captioning API')
app.register_blueprint(blueprint)

# namespace
ns = api.namespace('api', description='caption images')

# instantiate model
mt = model_trainer.ModelTrainer(training=False)
# indicate what version model to load
mt.load_model('ckpt-35')

"""
Routes
"""


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        image = request.files['file']
        if image.filename != '':
            # add file verification
            image_name = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join('static', image_name)
            image.save(image_path)
            caption = model_driver.get_prediction(image_path, mt.ckpt)
            result = {
                'caption': caption[0],
                'image_path': image_path
            }
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
                           help="image can't be blank and must be valid image type")
        parse.add_argument('candidates',
                           type=int,
                           default=0,
                           choices=list(range(11)),
                           help='number of candidate captions, from 0 to 10')

        args = parse.parse_args()

        image = args['file']
        image_name = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join('static', image_name)
        image.save(image_path)

        captions = model_driver.get_prediction(image_path,
                                               mt.ckpt,
                                               args['candidates'])

        result = {
            'captions': captions
        }

        return result


if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5001, debug=True)
