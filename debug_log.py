"""
Debug logging for World Space Onion Skin addon.
Writes to a log file in the addon directory for debugging.
"""

import os
import time
from datetime import datetime

# Log file path (same directory as this module)
_LOG_FILE = os.path.join(os.path.dirname(__file__), "debug.log")
_ENABLED = False  # DISABLED for performance - logging causes disk I/O in draw loop!
_MAX_LINES = 500  # Keep last N lines to prevent huge files

def _get_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(message, category="INFO"):
    """Write a log message to the debug file."""
    if not _ENABLED:
        return

    try:
        line = f"[{_get_timestamp()}] [{category}] {message}\n"
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # Silently fail if can't write

def log_frame_change(frame, gp_name, baked_offset, z_offset_setting, final_z):
    """Log frame change event with all relevant values."""
    log(f"FRAME={frame} GP={gp_name} baked_offset={baked_offset:.4f} z_setting={z_offset_setting:.4f} final_z={final_z:.4f}", "FRAME")

def log_onion_draw(current_frame, onion_frame, z_offset, stroke_count):
    """Log onion skin drawing event."""
    log(f"DRAW current={current_frame} onion={onion_frame} z_offset={z_offset:.4f} strokes={stroke_count}", "ONION")

def log_bake(frame_count, offset_range):
    """Log bake operation."""
    log(f"BAKE frames={frame_count} offset_range={offset_range}", "BAKE")

def log_cache(frame, stroke_count, is_bake_valid):
    """Log cache operation."""
    log(f"CACHE frame={frame} strokes={stroke_count} bake_valid={is_bake_valid}", "CACHE")

def log_cursor(frame, loc_z, delta_z, matrix_z, expected_offset, cursor_z):
    """Log cursor update values for debugging intermittent cursor/canvas issues."""
    log(f"frame={frame} loc.z={loc_z:.4f} delta_z={delta_z:.4f} matrix_z={matrix_z:.4f} expected={expected_offset:.4f} cursor_z={cursor_z:.4f}", "CURSOR")

def log_canvas(frame, success, stroke_placement=None, error_msg=None):
    """Log canvas alignment result for debugging."""
    if success:
        log(f"frame={frame} canvas=OK placement={stroke_placement}", "CANVAS")
    else:
        log(f"frame={frame} canvas=FAILED error={error_msg}", "CANVAS")

def log_error(message):
    """Log an error."""
    log(message, "ERROR")

def clear_log():
    """Clear the log file."""
    try:
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Debug Log Started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception:
        pass

def get_log_path():
    """Return the path to the log file."""
    return _LOG_FILE

# Clear log on module load
clear_log()
