import argparse
import sys

import tensorflow as tf

# Give clear error messages.
if sys.version_info >= (3,0):
    raise ImportError(
        "The tensorflow-serving-api package is not yet available"
        "for python3")

IMPORT_ERROR_TEMPLATE = (
    "This sub command requires the `{package}` pip package "
    "please install it with: \n"
    "    pip install {package}")

try:
    from grpc.beta import implementations
except ImportError:
    raise ImportError(IMPORT_ERROR_TEMPLATE.format(package='grpcio'))

try:
    from tensorflow_serving.apis import prediction_service_pb2
    from tensorflow_serving.apis import predict_pb2
except ImportError:
    raise ImportError(
        IMPORT_ERROR_TEMPLATE.format(package='tensorflow-serving-api'))



parser = argparse.ArgumentParser()

parser.add_argument('--host', type=str, default='localhost',
                    help="The hostname of the model server.")

parser.add_argument('--port', type=int, default=8500,
                    help="Model server port number.")

EXAMPLE_FORMAT = 'examples'
TENSOR_FORMAT = 'tensors'
CSV_FORMAT = 'csv'
parser.add_argument('--receiver_format', type=str, default='examples',
                    choices=[EXAMPLE_FORMAT, TENSOR_FORMAT, CSV_FORMAT])

def encode_float_dict(example_dict):
    """Serialize a {string:float} dict as a tf.train.Example."""
    # Convert each value into tf.train.Feature
    my_feature_dict = {}
    for key, value in example_dict.items():
        value = tf.train.FloatList(value=[value])
        my_feature_dict[key] = tf.train.Feature(float_list=value)

    # Convert the dict to a tf.train.Example.
    features = tf.train.Features(feature=my_feature_dict)
    example = tf.train.Example(features=features)

    # Return the example, serialize into a string.
    return example.SerializeToString()

def encode_examples(x):
    """Encode a series of examples as tf.train.Examples."""
    for index, row in pd.DataFrame(x).iterrows():
        yield encode_float_dict(dict(row))


def encode_csv(row):
    row = row.astype(str)
    return ','.join(row)

def prediction_client(args):
    """Send the iris test data to the tensorflow_model_server."""
    # Load and encode the data
    (train_x,train_y), (test_x, test_y) = iris_data.load_data()
    del train_x, train_y, test_y

    encoded_x = list(encode_examples(test_x))

    # Create a stub connected to host:port.
    channel = implementations.insecure_channel(args.host, args.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Create a prediction request.
    request = predict_pb2.PredictRequest()

    # Use the `predict` method of the model named `serve`.
    request.model_spec.name = 'serve'
    request.model_spec.signature_name = 'predict'

    # Calculate the class_ids and probabilities.
    request.output_filter.append('class_ids')
    request.output_filter.append('probabilities')

    if args.receiver_format == EXAMPLE_FORMAT:

    elif args.receiver_format == TENSOR_FORMAT:

    else:
        assert args.receiver_format == CSV_FORMAT

    # Attach the examples to the request.
    request.inputs['examples'].CopyFrom(tf.make_tensor_proto(encoded_x))

    # Send the request.
    timeout_seconds = 5.0
    future = stub.Predict.future(request, timeout_seconds)

    # Wait for the result.
    result = future.result()

    # Convert the result to a dictionary of numpy arrays.
    result = dict(result.outputs)
    result = {key: tf.make_ndarray(value) for key, value in result.items()}

    # Print the results
    print("\nClass_ids:")
    print(result['class_ids'])

    print("\nProbabilities for each class:")
    print(result['probabilities'])