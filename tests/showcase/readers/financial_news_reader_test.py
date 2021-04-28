# Copyright 2021 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for FinancialNewsReader.
"""
import os
import tempfile
import unittest
from ddt import ddt, data
from forte.data.data_pack import DataPack
from forte.data.span import Span
from forte.pipeline import Pipeline
from composable_source.readers import FinancialNewsReader


@ddt
class FinancialNewsReaderTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.orig_text = "-- The redundant line \n The Original News Text Here"
        self.processed_text = "The Original News Text Here"
        file_path = os.path.join(self.test_dir.name, 'test.html')
        with open(file_path, 'w') as f:
            f.write(self.orig_text)

    def tearDown(self):
        # Remove the directory after the test
        self.test_dir.cleanup()

    def test_reader_no_replace_test(self):
        # Read with no replacements
        pipeline = Pipeline()
        reader = FinancialNewsReader()
        pipeline.set_reader(reader, {'file_ext': '.html'})
        pipeline.initialize()

        pack = pipeline.process_one(self.test_dir.name)
        self.assertEqual(pack.text, self.processed_text)

    @data(
        # No replacement
        ([], 'The Original News Text Here'),
        # Insertion
        ([(Span(4, 4), 'New ')], 'The New Original News Text Here'),
        # Single, sorted multiple and unsorted multiple replacements
        ([(Span(4, 12), 'New')], 'The New News Text Here'),
        ([(Span(0, 4), ''), (Span(17, 22), '')], 'Original News Here'),
        ([(Span(17, 22), ''), (Span(0, 4), '')], 'Original News Here'),
    )
    def test_reader_replace_back_test(self, value):
        # Reading with replacements - replacing a span and changing it back
        span_ops, output = value

        pipeline = Pipeline()
        reader = FinancialNewsReader()
        reader.text_replace_operation = lambda _: span_ops
        pipeline.set_reader(reader, {'file_ext': '.html'})
        pipeline.initialize()

        pack: DataPack = pipeline.process_one(self.test_dir.name)
        self.assertEqual(pack.text, output)

    @data(
        # before span starts
        (Span(0, 3), Span(0, 3), "relaxed"),
        (Span(0, 3), Span(0, 3), "strict"),
        # after span ends
        (Span(8, 15), Span(12, 14), "relaxed"),
        # span itself
        (Span(4, 7), Span(4, 12), "relaxed"),
        # complete string
        (Span(0, 32), Span(0, 26), "strict"),
        # cases ending to or starting from between the span
        (Span(4, 32), Span(4, 26), "relaxed"),
        (Span(5, 32), Span(4, 26), "relaxed"),
        (Span(7, 32), Span(12, 26), "relaxed"),
        (Span(5, 32), Span(4, 26), "backward"),
        (Span(5, 32), Span(12, 26), "forward"),
        (Span(0, 5), Span(0, 12), "relaxed"),
        (Span(0, 6), Span(0, 4), "backward"),
        (Span(0, 7), Span(0, 12), "forward"),
        # same begin and end
        (Span(30, 30), Span(24, 24), "relaxed"),
        (Span(30, 30), Span(24, 24), "strict"),
        (Span(30, 30), Span(24, 24), "backward"),
        (Span(30, 30), Span(24, 24), "forward")
    )
    def test_reader_original_span_test(self, value):
        span_ops, output = ([(Span(4, 12), 'New'),
                             (Span(12, 13), ' Shiny '),
                             (Span(22, 22), ' Ends')],
                            'The New Shiny News Text Ends Here')
        input_span, expected_span, mode = value

        pipeline = Pipeline()
        reader = FinancialNewsReader()
        reader.text_replace_operation = lambda _: span_ops
        pipeline.set_reader(reader, {'file_ext': '.html'})
        pipeline.initialize()

        pack = pipeline.process_one(self.test_dir.name)

        self.assertEqual(pack.text, output)

        output_span = pack.get_original_span(input_span, mode)
        self.assertEqual(output_span, expected_span,
                         f"Expected: ({expected_span.begin, expected_span.end}"
                         f"), Found: ({output_span.begin, output_span.end})"
                         f" when Input: ({input_span.begin, input_span.end})"
                         f" and Mode: {mode}")

    @data(
        ([(Span(5, 8), ''), (Span(6, 10), '')], None),  # overlap
        ([(Span(5, 8), ''), (Span(6, 1000), '')], None),  # outside limit
    )
    def test_reader_replace_error_test(self, value):
        # Read with errors in span replacements
        span_ops, output = value

        pipeline = Pipeline()
        reader = FinancialNewsReader()
        reader.text_replace_operation = lambda _: span_ops
        pipeline.set_reader(reader, {'file_ext': '.html'})
        pipeline.initialize()

        with self.assertRaises(ValueError):
            pipeline.process(self.test_dir.name)


if __name__ == '__main__':
    unittest.main()
