"""Strip filesystem paths from exception text before it reaches a client.

Invariant 5: a server path must never appear in a client-facing message.
Exception text is useful ("couldn't connect to huggingface.co") but routinely
embeds paths (the HF cache, a temp upload), so it is redacted rather than
dropped. The unredacted exception still goes to the server log.
"""
from __future__ import annotations

import re

# Windows drive paths, UNC paths, and POSIX paths of two or more segments.
_PATH = re.compile(
    # The lookbehind stops "https://" reading as drive "s:/".
    r"""(?:(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\)[^\s'"<>|]+"""
    r"""|(?<![\w.:/])/(?:[^\s'"<>|/]+/)+[^\s'"<>|]*"""
)


def redact(text: object) -> str:
    return _PATH.sub("<path>", str(text))
