#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of DNNClassifier for Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()

DEFAULT_NPY_PATH = "inputs.npy"

parser.add_argument('--batch_size', default=100,
                    type=int, help='batch size')
parser.add_argument('--train_steps', default=200,
                    type=int, help='number of training steps')
parser.add_argument('--export_dir', type=str,
                    default="saved_model_exports",
                    help="Base directory to export the model to.")

EXAMPLE_FORMAT = 'examples'
TENSOR_FORMAT = 'tensors'
CSV_FORMAT = 'csv'
parser.add_argument('--receiver_format', type=str, default='examples',
                    choices=[EXAMPLE_FORMAT, TENSOR_FORMAT, CSV_FORMAT])


def csv_receiver(features):
   def my_receiver():
       # The placeholder is where the parent program will write its input.
       csv_input = tf.placeholder(tf.string, (None,))

       feature_keys = features.keys()
       csv_defaults = [[0.0]]*(len(feature_keys))

       # Build the feature dictionary from the parsed csv string.
       parsed_fields = tf.decode_csv(csv_input, csv_defaults)
       my_features = {}
       for key, field in zip(feature_keys, parsed_fields):
           my_features[key] = field

       # return the two pipeline ends in a ServingInputReceiver
       return tf.estimator.export.ServingInputReceiver(
           my_features, csv_input)

   return my_receiver

def float_dict_feature_spec(features):
    my_feature_spec = {}
    for key in features.keys():
        my_feature_spec[key] = tf.FixedLenFeature((), tf.float32)

    return my_feature_spec


def main(argv):
    """Train a simple model and export it as a Saved Model"""
    args = parser.parse_args(argv[1:])

    # Fetch the datasets
    train, test = iris_data.load_data()
    features, labels = train

    # Feature columns describe the input: all columns are numeric.
    feature_columns = [tf.feature_column.numeric_column(col_name)
                       for col_name in features.keys()]

    # Build DNN with 10, 10 units.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3)

    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(features, labels,
                                                  batch_size = 100),
        steps=args.train_steps)

    # Build the receiver function.


    if args.receiver_format == EXAMPLE_FORMAT:
        my_feature_spec = float_dict_feature_spec(dict(features))

        my_receiver = (
            tf.estimator.export.
                build_parsing_serving_input_receiver_fn(my_feature_spec))

    elif args.receiver_format == TENSOR_FORMAT:
        # its strange that this only works on tensors.
        feature_batch  = iris_data.eval_input_fn(dict(features), labels=None,
                                                 batch_size = 100)
        my_receiver = (
            tf.estimator.export
                .build_raw_serving_input_receiver_fn(feature_batch))

    else:
        assert args.receiver_format == CSV_FORMAT
        my_receiver = csv_receiver(features)

    # Build the Saved Model.
    savedmodel_path = classifier.export_savedmodel(
        export_dir_base=os.path.join(args.export_dir, args.receiver_format),
        serving_input_receiver_fn=my_receiver)

    print(u"\nModel exported to:\n    " + savedmodel_path.decode())

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(sys.argv)
