#  Copyright 2021 The Forte Authors. All Rights Reserved.
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
Util functions for evaluators.
"""
from typing import List
from forte.data.ontology import Annotation


def count_exact_match(
    refer_tag: List[Annotation], pred_tag: List[Annotation], attribute: str
) -> int:
    count = 0
    seen = set()
    for tag in refer_tag:
        span = (tag.begin, tag.end)
        attr = getattr(tag, attribute)
        seen.add((span, attr))
    for tag in pred_tag:
        span = (tag.begin, tag.end)
        attr = getattr(tag, attribute)
        if (span, attr) in seen:
            count += 1
            seen.remove((span, attr))
    return count
