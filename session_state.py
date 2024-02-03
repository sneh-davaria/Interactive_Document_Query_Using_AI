import streamlit as st

class SessionState:
    def __init__(self, **kwargs):
        """A new SessionState object."""
        for key, val in kwargs.items():
            setattr(self, key, val)

def get(**kwargs):
    """Gets a SessionState object for the current session."""
    # Use st.session_state for Streamlit 1.15.0 and above
    if hasattr(st, "session_state"):
        return st.session_state

    # Fallback to the previous hack for older Streamlit versions
    ctx = st.report_thread.get_report_ctx()

    this_session = None

    for session_info in st.server.server._session_info_by_id.values():
        s = session_info.session
        if s._uploaded_file_mgr == ctx.uploaded_file_mgr:
            this_session = s
            break

    if this_session is None:
        raise RuntimeError(
            "Could not get Streamlit Session object. "
            "Are you using a compatible version of Streamlit?"
        )

    if not hasattr(this_session, "_custom_session_state"):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state
