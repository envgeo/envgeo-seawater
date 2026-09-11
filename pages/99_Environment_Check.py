from pathlib import Path
import runpy


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "env_check_streamlit.py"


runpy.run_path(str(TOOL_PATH), run_name="__main__")
