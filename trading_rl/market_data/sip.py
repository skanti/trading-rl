"""Shared timing rules for historical SIP requests."""

from datetime import timedelta

# Historical SIP access excludes the most recent 15 minutes on the basic plan.
# Five extra minutes keep requests clear of the boundary and clock skew.
RECENT_SIP_SAFETY_DELAY = timedelta(minutes=20)
