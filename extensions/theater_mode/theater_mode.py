"""
Theater Mode Extension
Enables a fullscreen "theater mode" to view the video preview without distractions.
"""
from PySide6 import QtWidgets

class TheaterModeExtension:
    def __init__(self, main_window):
        self.main_window = main_window
        self.toggle_button = None

    def toggle_theater_mode(self):
        """Toggles the main window between fullscreen and normal windowed mode."""
        if self.main_window.isFullScreen():
            self.main_window.showNormal()
            if self.toggle_button:
                self.toggle_button.setText("Enter Theater Mode")
        else:
            self.main_window.showFullScreen()
            if self.toggle_button:
                self.toggle_button.setText("Exit Theater Mode")

    def on_app_close(self):
        """Safety hook: ensures the app exits fullscreen if it's closed while in theater mode."""
        if self.main_window.isFullScreen():
            self.main_window.showNormal()

def setup(main_window):
    # 1. Initialize the extension class
    ext = TheaterModeExtension(main_window)
    
    # 2. Create the UI tab content
    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)
    
    # Add a descriptive label (word-wrapped for the narrow right panel)
    label = QtWidgets.QLabel(
        "Toggle fullscreen theater mode for an immersive, distraction-free viewing experience."
    )
    label.setWordWrap(True)
    layout.addWidget(label)
    
    # Add the toggle button
    ext.toggle_button = QtWidgets.QPushButton("Enter Theater Mode")
    ext.toggle_button.clicked.connect(ext.toggle_theater_mode)
    layout.addWidget(ext.toggle_button)
    
    # Push content to the top
    layout.addStretch()
    
    # 3. Add the tab to the main UI using the supported helper method
    # Title is kept short ("Theater") so it doesn't take up too much space in the top tab row
    main_window.add_parameter_tab(tab, "Theater")
    
    # 4. Return the extension instance
    return ext