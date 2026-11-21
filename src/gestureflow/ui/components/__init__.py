"""
gestureflow.ui.components
~~~~~~~~~~~~~~~~~~~~~~~~~
Export design tokens, reusable cards, charts, sidebar, and onboarding dialog.
"""

from gestureflow.ui.components.theme import (
    BG_DARK,
    SIDEBAR_BG,
    CARD_BG,
    CARD_BG_HOVER,
    BORDER_COLOR,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    DANGER_RED,
    WARNING_AMBER,
    get_application_stylesheet,
)
from gestureflow.ui.components.cards import (
    StatusCard,
    MetricCard,
    GestureCard,
    SectionHeader,
)
from gestureflow.ui.components.sidebar import SidebarNavigation
from gestureflow.ui.components.charts import RollingMetricChart
from gestureflow.ui.components.onboarding_dialog import OnboardingDialog

__all__ = [
    "BG_DARK",
    "SIDEBAR_BG",
    "CARD_BG",
    "CARD_BG_HOVER",
    "BORDER_COLOR",
    "ACCENT_EMERALD",
    "ACCENT_CYAN",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "DANGER_RED",
    "WARNING_AMBER",
    "get_application_stylesheet",
    "StatusCard",
    "MetricCard",
    "GestureCard",
    "SectionHeader",
    "SidebarNavigation",
    "RollingMetricChart",
    "OnboardingDialog",
]
