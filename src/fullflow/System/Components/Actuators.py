from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fullflow.System import Component

if TYPE_CHECKING:
    from fullflow.System import Network, State



class FirstOrderActuator(Component): pass


class RateLimitedActuator(Component): pass