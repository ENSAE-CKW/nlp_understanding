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

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node

from .nodes_cnn_char import cnn_test

def create_cnn_char_test(**kwargs):
    return Pipeline(
        [
            node(
                func= cnn_test
                , inputs= ["test_data", "params:cnn_cuda_allow", "params:cnn_size_batch"
                     , "params:cnn_num_threads", "cnn_char_model", "params:cnn_feature_num","params:cnn_sequence_len"
                    , "params:cnn_feature_size", "params:cnn_kernel_one", "params:cnn_kernel_two", "params:cnn_stride_one"
                    , "params:cnn_stride_two", "params:cnn_output_linear", "params:cnn_num_class", "params:cnn_dropout"
                    , "params:cnn_type_map", "params:cnn_seuil" ,"params:cnn_type_agg"]
                , outputs= None
                , tags= ["cnn_char_test", "test"]
            )
        ]
    )