"""
Hotkey management for CLI.

Thin wrapper around the `keyboard` library for system-wide hotkeys.
Provides predictable, minimal behavior suitable for small ML/CLI tools.
"""

import keyboard
from typing import Callable, Dict


class HotkeyManager:
    """
    Manage global hotkey bindings.

    Attributes:
        bindings (Dict[str, Callable]): Registered hotkey → callback map.
    """

    def __init__(self) -> None:
        """Initialize an empty hotkey registry."""
        self.bindings: Dict[str, Callable] = {}

    def bind(self, combo: str, callback: Callable) -> None:
        """
        Register a system-wide hotkey.

        Args:
            combo (str): Key combo (e.g. "ctrl+shift+s").
            callback (Callable): Function to run when pressed.
        """
        self.bindings[combo] = callback
        keyboard.add_hotkey(combo, callback)

    def start(self) -> None:
        """
        Block and listen for hotkey events indefinitely.
        """
        keyboard.wait()
