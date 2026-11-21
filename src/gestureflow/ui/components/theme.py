"""
gestureflow.ui.components.theme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Central design system tokens and Qt stylesheets for GestureFlow.
Enforces a commercial-grade dark interface:
- Deep charcoal backgrounds (#0B1018, #111827)
- Rounded cards and subtle borders (#182231, #293548)
- Emerald green (#34D399) and cyan (#22D3EE) accents
- Accessible contrast typography (#F3F4F6, #9CA3AF)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
BG_DARK = "#0B1018"
SIDEBAR_BG = "#111827"
CARD_BG = "#182231"
CARD_BG_HOVER = "#1F2B3D"
CARD_BG_ACTIVE = "#243245"
BORDER_COLOR = "#293548"
BORDER_COLOR_LIGHT = "#374761"

ACCENT_EMERALD = "#34D399"
ACCENT_EMERALD_HOVER = "#10B981"
ACCENT_EMERALD_DIM = "rgba(52, 211, 153, 0.15)"

ACCENT_CYAN = "#22D3EE"
ACCENT_CYAN_HOVER = "#06B6D4"
ACCENT_CYAN_DIM = "rgba(34, 211, 238, 0.15)"

TEXT_PRIMARY = "#F3F4F6"
TEXT_SECONDARY = "#9CA3AF"
TEXT_MUTED = "#6B7280"

DANGER_RED = "#EF4444"
DANGER_RED_HOVER = "#DC2626"
DANGER_RED_DIM = "rgba(239, 68, 68, 0.15)"

WARNING_AMBER = "#F59E0B"
WARNING_AMBER_DIM = "rgba(245, 158, 11, 0.15)"

INFO_BLUE = "#3B82F6"


# ---------------------------------------------------------------------------
# Global Qt Stylesheet
# ---------------------------------------------------------------------------
def get_application_stylesheet() -> str:
    """Return the complete global Qt stylesheet for the application."""
    return f"""
    /* Core window and background */
    QMainWindow, QDialog {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        font-size: 13px;
    }}

    QWidget {{
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}

    /* Global tooltips */
    QToolTip {{
        background-color: {CARD_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        padding: 5px 9px;
        font-size: 12px;
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        background: {BG_DARK};
        width: 10px;
        margin: 0px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_COLOR};
        min-height: 25px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {BORDER_COLOR_LIGHT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {BG_DARK};
        height: 10px;
        margin: 0px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_COLOR};
        min-width: 25px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {BORDER_COLOR_LIGHT};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {CARD_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background-color: {CARD_BG_HOVER};
        border-color: {BORDER_COLOR_LIGHT};
    }}
    QPushButton:pressed {{
        background-color: {CARD_BG_ACTIVE};
    }}
    QPushButton:disabled {{
        background-color: rgba(24, 34, 49, 0.4);
        color: {TEXT_MUTED};
        border-color: rgba(41, 53, 72, 0.5);
    }}

    /* Accent & Action Buttons */
    QPushButton#btn_primary, QPushButton.primary {{
        background-color: {ACCENT_EMERALD};
        color: #064E3B;
        border: none;
    }}
    QPushButton#btn_primary:hover, QPushButton.primary:hover {{
        background-color: {ACCENT_EMERALD_HOVER};
    }}

    QPushButton#btn_danger, QPushButton.danger {{
        background-color: {DANGER_RED};
        color: #FFFFFF;
        border: none;
    }}
    QPushButton#btn_danger:hover, QPushButton.danger:hover {{
        background-color: {DANGER_RED_HOVER};
    }}

    /* Inputs & Combos */
    QComboBox {{
        background-color: {CARD_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 500;
    }}
    QComboBox:hover {{
        border-color: {BORDER_COLOR_LIGHT};
    }}
    QComboBox:focus {{
        border-color: {ACCENT_CYAN};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {SIDEBAR_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        selection-background-color: {ACCENT_EMERALD_DIM};
        selection-color: {ACCENT_EMERALD};
        padding: 4px;
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {CARD_BG};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_COLOR};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 12px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT_CYAN};
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        background: {BORDER_COLOR};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT_EMERALD};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: #FFFFFF;
        border: 2px solid {ACCENT_EMERALD};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {ACCENT_EMERALD};
        border-color: #FFFFFF;
    }}

    /* Checkboxes */
    QCheckBox {{
        spacing: 8px;
        color: {TEXT_PRIMARY};
        font-size: 12px;
        font-weight: 500;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        background: {CARD_BG};
    }}
    QCheckBox::indicator:hover {{
        border-color: {ACCENT_EMERALD};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT_EMERALD};
        border-color: {ACCENT_EMERALD};
        image: none;
    }}

    /* Labels & Headers */
    QLabel {{
        color: {TEXT_PRIMARY};
    }}
    QLabel#lbl_muted {{
        color: {TEXT_SECONDARY};
    }}

    /* TabWidget */
    QTabWidget::pane {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        background: {CARD_BG};
        padding: 8px;
    }}
    QTabBar::tab {{
        background: {SIDEBAR_BG};
        color: {TEXT_SECONDARY};
        padding: 8px 16px;
        margin-right: 4px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }}
    QTabBar::tab:selected {{
        background: {CARD_BG};
        color: {ACCENT_CYAN};
        border-bottom: 2px solid {ACCENT_CYAN};
    }}
    QTabBar::tab:hover:!selected {{
        background: {CARD_BG_HOVER};
        color: {TEXT_PRIMARY};
    }}

    /* Status bar */
    QStatusBar {{
        background: {SIDEBAR_BG};
        color: {TEXT_SECONDARY};
        border-top: 1px solid {BORDER_COLOR};
        font-size: 11px;
    }}

    /* Splitters */
    QSplitter::handle {{
        background: {BORDER_COLOR};
    }}
    """
