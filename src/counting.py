"""Finite-segment crossing geometry and per-track event de-duplication."""

from __future__ import annotations


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _side(line_a, line_b, point):
    return _cross((line_b[0] - line_a[0], line_b[1] - line_a[1]), (point[0] - line_a[0], point[1] - line_a[1]))


def segments_intersect(p0, p1, q0, q1):
    """Return true when finite segment p0-p1 crosses finite segment q0-q1."""
    r = (p1[0] - p0[0], p1[1] - p0[1])
    s = (q1[0] - q0[0], q1[1] - q0[1])
    denominator = _cross(r, s)
    if abs(denominator) < 1e-9:
        return False
    qmp = (q0[0] - p0[0], q0[1] - p0[1])
    t = _cross(qmp, s) / denominator
    u = _cross(qmp, r) / denominator
    return 0 <= t <= 1 and 0 <= u <= 1


class LineCounter:
    def __init__(self, line_a, line_b, debounce_frames=4):
        self.line_a = tuple(line_a)
        self.line_b = tuple(line_b)
        self.debounce_frames = debounce_frames
        self.previous = {}
        self.last_event_frame = {}
        self.counts = {"negative_to_positive": 0, "positive_to_negative": 0}

    def update(self, track_id, point, frame):
        point = tuple(point)
        old = self.previous.get(track_id)
        if old is None:
            self.previous[track_id] = point
            return None
        old_side, new_side = _side(self.line_a, self.line_b, old), _side(self.line_a, self.line_b, point)
        # Keep the last off-line point through a frame exactly on the line. That
        # lets the next frame determine whether the track crossed or bounced back.
        if new_side == 0:
            return None
        if old_side == 0:
            self.previous[track_id] = point
            return None
        if (old_side > 0) == (new_side > 0):
            self.previous[track_id] = point
            return None
        if not segments_intersect(old, point, self.line_a, self.line_b):
            self.previous[track_id] = point
            return None
        if frame - self.last_event_frame.get(track_id, -10**9) < self.debounce_frames:
            self.previous[track_id] = point
            return None
        direction = "negative_to_positive" if old_side < 0 < new_side else "positive_to_negative"
        self.counts[direction] += 1
        self.last_event_frame[track_id] = frame
        self.previous[track_id] = point
        return {"track_id": int(track_id), "frame": int(frame), "direction": direction, "x": point[0], "y": point[1]}
