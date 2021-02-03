# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.



from kedro.pipeline import Pipeline, node

from .nodes_logistic_bow import encode_data


def create_logistic_bow_pipeline_de(**kwargs):
    return Pipeline(
        [
            node(
                func= encode_data
                , inputs= ["allocine_train", "params:logistic_bow_max_number_feature"]
                , outputs= ["train_data_logistic_bow", "train_y_logistic_bow", "vocabulary_logistic_bow"]
                , name= "train_data_logistic_bow"
                , tags= ["logistic_bow", "logistic_bow_train", "train"]
            ),
            node(
                func=encode_data
                , inputs= ["allocine_valid", "params:logistic_bow_max_number_feature", "vocabulary_logistic_bow"]
                , outputs= ["valid_data_logistic_bow", "valid_y_logistic_bow"]
                , name= "valid_data_logistic_bow"
                , tags= ["logistic_bow", "logistic_bow_train", "train"]
            ),
            node(
                func=encode_data
                , inputs= ["allocine_test", "params:logistic_bow_max_number_feature", "vocabulary_logistic_bow"]
                , outputs= ["test_data_logistic_bow", "test_y_logistic_bow"]
                , name= "test_data_logistic_bow"
                , tags= ["logistic_bow", "logistic_bow_test", "test"]
            )
        ]
    )
