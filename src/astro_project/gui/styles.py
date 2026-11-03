DARK_MODE_CSS = """
<style>
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
    }
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 92vh;
    }
    div[data-testid="stSidebarUserContent"] > div:first-child {
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1 !important;
        min-height: 88vh;
    }
    div[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlock"] {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stSidebarUserContent"] div[data-testid="stElementContainer"]:has(.sidebar-statusbar-wrap) {
        margin-top: auto !important;
    }
    .sidebar-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fafc;
        text-align: center;
        letter-spacing: 0.03em;
        margin-top: -0.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #334155;
    }
    .sidebar-statusbar-wrap {
        margin-top: auto;
        padding-top: 0.6rem;
        border-top: 1px solid #1f2937;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    .top-statusbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        box-sizing: border-box;
        height: 34px;
        padding: 0 14px;
        background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 6px;
        font-size: 0.84rem;
        font-family: 'Segoe UI', Roboto, monospace;
        color: #f8fafc;
        margin-bottom: 0.6rem;
        white-space: nowrap;
        overflow: hidden;
    }
    .top-statusbar span {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .status-badge {
        background-color: #0f172a;
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 1px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    div[data-testid="stImage"] img {
        max-height: 48vh !important;
        width: auto !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        display: block !important;
        object-fit: contain !important;
    }
    div[data-testid="stModal"] {
        background-color: rgba(5, 8, 15, 0.65) !important;
        backdrop-filter: blur(4px) !important;
    }
    div[role="dialog"] {
        background-color: rgba(17, 24, 39, 0.94) !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.7) !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #3b0764 !important;
        border: 1px solid #7c3aed !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #4c1d95 !important;
        border: 1px solid #7c3aed !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
    }
    .sidebar-hw-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 6px 8px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        line-height: 1.25;
        color: #f8fafc;
    }
    .sidebar-hw-label {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 600;
        display: block;
    }
</style>
"""
