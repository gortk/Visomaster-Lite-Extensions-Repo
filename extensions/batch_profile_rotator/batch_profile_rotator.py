"""
Batch Profile Rotator Extension

Rotates through selected face-editor profiles across batch image (and video)
jobs. Each processed media file uses the next ticked profile, wrapping at the
end of the list. Works for both "Batch Process Selected Face" and
"Batch Process All Faces" modes.

Enable the extension, tick the profiles in the tab, then start a batch job.
The status label shows which profile is currently being used.
"""

import functools
import threading
from PySide6 import QtWidgets, QtGui, QtCore

from app.ui.widgets.actions import face_editor_profile_actions
from app.ui.widgets.actions import common_actions as ca

_SETTINGS_ORG = "VisoMasterExtensions"
_SETTINGS_APP = "BatchProfileRotator"

class BatchProfileRotator:
    def __init__(self, main_window):
        self.main_window = main_window
        self.enabled = False
        self.profile_list = []
        self._index = -1
        self._last_media_path = None
        self._orig_process_current_frame = None
        self._status_label = None
        self._applying = False
        self._settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    # ---------- helpers ----------
    def _update_status(self, message):
        if self._status_label is not None:
            try:
                self._status_label.setText(message)
            except RuntimeError:
                self._status_label = None

    def _load_profile_data(self, profile_name):
        if profile_name == face_editor_profile_actions.DEFAULT_PROFILE_NAME:
            return {
                key: self.main_window.default_parameters[key]
                for key in face_editor_profile_actions.FACE_EDITOR_SETTINGS_KEYS
                if key in self.main_window.default_parameters
            }
        return face_editor_profile_actions._load_profile(profile_name)

    def _apply_profile_to_face(self, face_id, profile_name):
        try:
            data = self._load_profile_data(profile_name)
        except Exception as e:
            print(f"[Rotator] Failed to load profile '{profile_name}': {e}")
            return False

        prev_selected = self.main_window.selected_target_face_id
        prev_batch = getattr(self.main_window, "_batch_update_in_progress", False)
        self.main_window.selected_target_face_id = face_id
        self.main_window._batch_update_in_progress = True
        try:
            face_editor_profile_actions._apply_settings(self.main_window, data)
        except Exception as e:
            print(f"[Rotator] Failed to apply '{profile_name}' to face {face_id}: {e}")
            return False
        finally:
            self.main_window.selected_target_face_id = prev_selected
            self.main_window._batch_update_in_progress = prev_batch
        return True

    def _apply_current_profile(self):
        if not self.profile_list:
            return
        profile = self.profile_list[self._index % len(self.profile_list)]
        for face_id in list(self.main_window.target_faces.keys()):
            self._apply_profile_to_face(face_id, profile)

        # Refresh UI to reflect the (still-selected) target face
        if self.main_window.selected_target_face_id:
            try:
                ca.set_widgets_values_using_face_id_parameters(
                    self.main_window, self.main_window.selected_target_face_id
                )
            except Exception:
                pass

        self._update_status(
            f"Rotating: {profile} "
            f"({(self._index % len(self.profile_list)) + 1} of {len(self.profile_list)})"
        )

    def _load_checked(self):
        value = self._settings.value("checked_profiles", None)
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        return list(value) if value else []
    
    def _save_checked(self, names):
        self._settings.setValue("checked_profiles", list(names))
        self._settings.sync()
    
    def _load_enabled(self):
        return bool(self._settings.value("enabled", False, type=bool))
    
    def _save_enabled(self, value):
        self._settings.setValue("enabled", bool(value))
        self._settings.sync()
    

    # ---------- hook ----------
    def _on_process_current_frame(self, *args, **kwargs):
        if (
            self.enabled
            and not self._applying
            and getattr(self.main_window, "is_batch_processing", False)
            and threading.current_thread() is threading.main_thread()
            and self.profile_list
        ):
            self._applying = True
            prev_batch = getattr(self.main_window, "_batch_update_in_progress", False)
            self.main_window._batch_update_in_progress = True
            try:
                media_path = self.main_window.video_processor.media_path
                if media_path and media_path != self._last_media_path:
                    self._last_media_path = media_path
                    self._index += 1
                if self._last_media_path:
                    self._apply_current_profile()
            finally:
                self.main_window._batch_update_in_progress = prev_batch
                self._applying = False
        return self._orig_process_current_frame(*args, **kwargs)

    def _install_hook(self):
        if self._orig_process_current_frame is not None:
            return
        vp = self.main_window.video_processor
        self._orig_process_current_frame = vp.process_current_frame
        vp.process_current_frame = functools.partial(
            self._on_process_current_frame
        )

    # ---------- public ----------
    def set_enabled(self, checked):
        self.enabled = bool(checked)
        self._save_enabled(self.enabled)
        if self.enabled:
            self._install_hook()
            self._index = -1
            self._last_media_path = None
            self._update_status("Enabled — waiting for batch start")
        else:
            self._update_status("Disabled")

    def on_app_close(self):
        try:
            self.enabled = False
        except Exception:
            pass


def setup(main_window):
    ext = BatchProfileRotator(main_window)

    tab = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(tab)

    info = QtWidgets.QLabel(
        "<b>Rotate face-editor profiles across batch images.</b><br><br>"
        "1. Tick the profiles below.<br>"
        "2. Enable the rotator.<br>"
        "3. Start a batch job (Batch Process Selected Face or All Faces).<br><br>"
        "Each processed image (or video) uses the next ticked profile, wrapping "
        "at the end of the list. The <i>none</i> profile resets face-editor "
        "settings to defaults for that image.<br><br>"
        "<i>Note:</i> profiles apply to every target face present on that "
        "media. Selected-face mode reuses the current target faces; all-faces "
        "mode re-detects them per image."
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    enable_toggle = QtWidgets.QCheckBox("Enable Profile Rotation")
    layout.addWidget(enable_toggle)

    profile_group = QtWidgets.QGroupBox("Profiles to rotate")
    profile_layout = QtWidgets.QVBoxLayout(profile_group)
    profile_list = QtWidgets.QListWidget()
    profile_list.setSelectionMode(
        QtWidgets.QAbstractItemView.SelectionMode.NoSelection
    )
    profile_layout.addWidget(profile_list, 1)

    btn_row = QtWidgets.QHBoxLayout()
    refresh_btn = QtWidgets.QPushButton("Refresh List")
    select_all_btn = QtWidgets.QPushButton("Select All")
    clear_btn = QtWidgets.QPushButton("Clear Selection")
    btn_row.addWidget(refresh_btn)
    btn_row.addWidget(select_all_btn)
    btn_row.addWidget(clear_btn)
    profile_layout.addLayout(btn_row)
    layout.addWidget(profile_group, 1)

    status_label = QtWidgets.QLabel("Disabled")
    status_label.setStyleSheet("color: #888;")
    layout.addWidget(status_label)
    ext._status_label = status_label

    def refresh_profiles():
        saved = ext._load_checked()  # None = first run → check everything
        profile_list.blockSignals(True)
        try:
            profile_list.clear()
            try:
                names = face_editor_profile_actions.get_profile_names()
            except Exception as e:
                print(f"[Rotator] Could not list profiles: {e}")
                return
            for name in names:
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                checked = True if saved is None else (name in saved)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if checked
                    else QtCore.Qt.CheckState.Unchecked
                )
                profile_list.addItem(item)
        finally:
            profile_list.blockSignals(False)
        gather_selection(save=False)
    
    def gather_selection(save=True):
        selected = []
        for i in range(profile_list.count()):
            item = profile_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.text())
        ext.profile_list = selected
        if save:
            ext._save_checked(selected)
        if not ext.enabled:
            status_label.setText(
                f"Selected {len(selected)} profile(s)"
                if selected else "No profiles selected"
            )

    profile_list.itemChanged.connect(lambda _item: gather_selection())
    refresh_btn.clicked.connect(refresh_profiles)
    select_all_btn.clicked.connect(
        lambda: [
            profile_list.item(i).setCheckState(QtCore.Qt.CheckState.Checked)
            for i in range(profile_list.count())
        ]
    )
    clear_btn.clicked.connect(
        lambda: [
            profile_list.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
            for i in range(profile_list.count())
        ]
    )
    enable_toggle.toggled.connect(ext.set_enabled)

    refresh_profiles()
    if ext._load_enabled():
        enable_toggle.blockSignals(True)
        enable_toggle.setChecked(True)
        enable_toggle.blockSignals(False)
        ext.set_enabled(True)
    gather_selection()

    main_window.tabWidget.addTab(tab, "Profile Rotator")
    return ext