"""Reviewable automatic shared-boundary matching."""

from uuid import uuid4

from vectrify.document import StaleRevisionError
from vectrify.ui.simplify import render_previews


class ContactPreview:
    def __init__(self, session, payload):
        session.check_revision(payload)
        self.id = uuid4().hex
        self.epoch = session.epoch
        self.transaction = session.editor.transaction("Share boundaries")
        count = self.transaction.share_boundaries(float(payload.get("tolerance", 1)))
        self.result = {
            "id": self.id,
            "edges": count,
            "previews": render_previews(
                session.editor.snapshot.document,
                self.transaction.preview,
                payload.get("bounds", session.state(svg=False)["bounds"]),
                highlight=True,
            ),
        }

    def apply(self, session):
        if self.epoch != session.epoch:
            raise StaleRevisionError("The drawing changed. Preview boundaries again.")
        self.transaction.commit()
        return session.state()
