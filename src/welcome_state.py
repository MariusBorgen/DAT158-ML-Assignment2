"""Velkomsthendelser uten kamera eller modell; klokkeslett sendes inn av kalleren."""
class WelcomeState:
    def __init__(self, confirmations=3, absence_seconds=3.0):
        self.confirmations = confirmations
        self.absence_seconds = absence_seconds
        self.candidate = None
        self.count = 0
        self.last_face = None
        self.greeted = set()

    def observe(self, label, now, face_present=True):
        if self.last_face is not None and now - self.last_face >= self.absence_seconds:
            self.greeted.clear()
            self.candidate, self.count = None, 0
        if face_present:
            self.last_face = now
        if label is None:
            self.candidate, self.count = None, 0
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        if self.count >= self.confirmations and label not in self.greeted:
            self.greeted.add(label)
            return label
        return None
