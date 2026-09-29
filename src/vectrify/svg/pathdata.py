"""Shared tokenizer for SVG path data.

Both the mutation operators and the gradient fit walk `d` attributes command by
command, and a second copy of this pattern is a second place for the arc-flag
and leading-dot cases to be got wrong.
"""

import re

# Path data writes numbers with a leading dot and no separators ("M.5.5"), so
# an anchorless \d+ pattern would match the middle of a coordinate pair.
PATH_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?(?:\d+\.\d+|\.\d+|\d+))")
