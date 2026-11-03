"""Solution records and their restorable operating points."""


class SolutionRecords(list):
    """List of output records with in-memory operating-point history."""

    def __init__(self, records=()):
        super().__init__()
        self.operating_points = {}
        self.add_records(records)

    def add_records(self, records, time_value=None):
        """Append exported values without retaining mutable model arrays."""
        from .linearization.checkpoints import _copy

        rows = []
        for record in records:
            row = dict(record)
            row["value"] = _copy(record["value"])
            if time_value is not None:
                row["time"] = float(time_value)
            rows.append(row)
        self.extend(rows)
        return rows

    def capture(self, network, cache, issue=None):
        """Save the current model state for operating-point analysis."""
        from .linearization.checkpoints import OperatingCheckpoint

        point = OperatingCheckpoint(network, cache)
        self.operating_points[point.time] = (point, issue)
        return point
