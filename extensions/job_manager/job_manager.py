"""
Job Manager extension for VisoMaster Fusion
===========================================

Drop this file into the ``extensions/`` folder (next to the app root) and restart.
A "Jobs" tab appears in the tab row below the menu bar.

What it does
------------
* Preconfigure jobs: set a target up exactly the way you want it (target faces,
  assigned source faces / embeddings, per-face parameters, markers, segments,
  dropped frames and every global setting incl. the output folder), then press
  "Add current target as job". Repeat for as many targets as you like.
* Run the whole queue as one unattended batch (e.g. overnight). Every job
  restores its own snapshot, records the target, waits until the file is
  written, then moves on to the next job.
* Jobs are stored on disk (``job_manager_data/``), so the queue survives
  restarts and crashes.

Unattended / "headless" runs
----------------------------
* Start the app with ``--run-jobs`` (or set the environment variable
  ``VISOMASTER_RUN_JOBS=1``) and the pending jobs start by themselves once the
  workspace has finished loading. Combine it with "When finished: Close
  VisoMaster / Sleep / Shut down" and a Windows Task Scheduler entry.
* Optional: keep the PC awake while jobs run, auto-dismiss pop-up dialogs so a
  stray message box can never stall the queue, per-job time limit, retries,
  continue-after-failure.
* Everything is logged to ``job_manager_data/job_manager.log``.

Notes
-----
* Job snapshots are stored as pickle files in ``job_manager_data/jobs``. They are
  written and read only by this extension; do not load job files from people you
  do not trust.
* Denoiser reference K/V maps are not stored in a job; they are rebuilt from the
  source faces / embeddings that are loaded in the app when the job runs.
* While the queue is running the app is driven by the extension: do not click
  around in the main window.
"""

import copy
import ctypes
import gc
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

DATA_DIR_NAME = "job_manager_data"
SNAPSHOT_VERSION = 1

ST_PENDING = "pending"
ST_RUNNING = "running"
ST_DONE = "done"
ST_FAILED = "failed"

STATUS_LABEL = {
    ST_PENDING: "Pending",
    ST_RUNNING: "Running",
    ST_DONE: "Done",
    ST_FAILED: "Failed",
}

AFTER_ACTIONS = [
    ("none", "Do nothing"),
    ("close", "Close VisoMaster"),
    ("sleep", "Put the PC to sleep"),
    ("hibernate", "Hibernate the PC"),
    ("shutdown", "Shut the PC down (60 s delay)"),
]

DEFAULT_SETTINGS = {
    "continue_on_error": True,
    "retries": 0,
    "max_minutes": 0,
    "keep_awake": True,
    "dismiss_popups": True,
    "auto_start_on_launch": False,
    "after_action": "none",
}

OUTPUT_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".png", ".jpg", ".jpeg")


class JobError(Exception):
    """A problem with one job (shown in the job list and the log)."""


# --------------------------------------------------------------------------- #
# Lazy access to the app modules (keeps this file importable for tests)
# --------------------------------------------------------------------------- #
_APP = None


def _app():
    global _APP
    if _APP is None:
        import app.helpers.miscellaneous as misc_helpers
        from app.ui.widgets import ui_workers, widget_components
        from app.ui.widgets.actions import (
            card_actions,
            control_actions,
            list_view_actions,
            save_load_actions,
            video_control_actions,
        )
        from app.ui.widgets.actions import common_actions as common_widget_actions

        _APP = SimpleNamespace(
            misc_helpers=misc_helpers,
            ui_workers=ui_workers,
            widget_components=widget_components,
            card_actions=card_actions,
            control_actions=control_actions,
            list_view_actions=list_view_actions,
            save_load_actions=save_load_actions,
            video_control_actions=video_control_actions,
            common_widget_actions=common_widget_actions,
        )
    return _APP


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _norm(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _plain(params):
    """ParametersDict / dict -> plain deep-copied dict."""
    if params is None:
        return None
    data = params.data if hasattr(params, "data") else params
    return copy.deepcopy(dict(data))


def _copy_store(store):
    return {str(m): np.array(e, copy=True) for m, e in (store or {}).items()}


def _atomic_write_bytes(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def _set_keep_awake(enabled: bool, holder: dict):
    """Keep the PC from going to sleep while jobs run (best effort)."""
    try:
        if sys.platform.startswith("win"):
            es_continuous = 0x80000000
            es_system_required = 0x00000001
            flags = es_continuous | (es_system_required if enabled else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        elif sys.platform == "darwin":
            proc = holder.pop("caffeinate", None)
            if proc is not None:
                proc.terminate()
            if enabled:
                holder["caffeinate"] = subprocess.Popen(
                    ["caffeinate", "-i", "-w", str(os.getpid())]
                )
    except Exception as exc:
        print(f"[JOB MANAGER] Could not change keep-awake state: {exc}")


# --------------------------------------------------------------------------- #
# Snapshot capture / restore
# --------------------------------------------------------------------------- #
def capture_snapshot(mw) -> dict:
    """Collect everything that defines how the *current* target is processed."""
    app = _app()
    wc = app.widget_components
    sla = app.save_load_actions

    card = mw.selected_video_button
    if not isinstance(card, wc.TargetMediaCardButton):
        raise JobError("Select a target media first.")

    vp = mw.video_processor
    merged = app.list_view_actions.get_all_merged_embeddings(mw)

    faces = []
    for face_id, btn in mw.target_faces.items():
        crop = btn.cropped_face
        if hasattr(crop, "cpu"):
            crop = crop.cpu().numpy()
        crop = np.ascontiguousarray(np.asarray(crop)).copy()

        assigned_inputs = {}
        for input_id, store in btn.assigned_input_faces.items():
            live = mw.input_faces.get(input_id)
            assigned_inputs[str(input_id)] = {
                "media_path": getattr(live, "media_path", None),
                "embedding_store": _copy_store(store),
            }
        assigned_embeddings = {}
        for emb_id, store in btn.assigned_merged_embeddings.items():
            live = merged.get(emb_id)
            assigned_embeddings[str(emb_id)] = {
                "name": getattr(live, "embedding_name", None),
                "embedding_store": _copy_store(store),
            }

        faces.append(
            {
                "face_id": str(face_id),
                "cropped_face": crop,
                "embedding_store": _copy_store(btn.embedding_store),
                "assigned_input_faces": assigned_inputs,
                "assigned_merged_embeddings": assigned_embeddings,
                "parameters": _plain(mw.parameters.get(face_id)),
            }
        )

    markers = sla.sanitize_markers_dictionary(
        mw,
        sla.convert_markers_to_supported_type(mw, copy.deepcopy(mw.markers), dict),
    )

    segments = []
    for s in getattr(mw, "segments", []):
        params = s.get("parameters")
        segments.append(
            {
                "start": int(s["start"]),
                "end": int(s["end"]),
                "kind": s.get("kind", "normal"),
                "parameters": (
                    {str(fid): _plain(pd) for fid, pd in params.items()}
                    if params is not None
                    else None
                ),
                "control": copy.deepcopy(s.get("control"))
                if s.get("control") is not None
                else None,
            }
        )

    return {
        "version": SNAPSHOT_VERSION,
        "media_path": card.media_path,
        "file_type": card.file_type,
        "control": copy.deepcopy(dict(mw.control)),
        "swap_faces_enabled": bool(mw.swapfacesButton.isChecked()),
        "edit_faces_enabled": bool(mw.editFacesButton.isChecked()),
        "selected_target_face_id": mw.selected_target_face_id,
        "current_widget_parameters": _plain(mw.current_widget_parameters),
        "target_faces": faces,
        "markers": markers,
        "segments": segments,
        "dropped_frames": sorted(int(f) for f in getattr(mw, "dropped_frames", [])),
        "max_frame_number": int(getattr(vp, "max_frame_number", 0) or 0),
    }


def restore_snapshot(mw, snap: dict, log=print):
    """Rebuild the app state of a snapshot (target media must already be loaded)."""
    app = _app()
    sla = app.save_load_actions
    vca = app.video_control_actions
    cwa = app.common_widget_actions
    lva = app.list_view_actions
    misc = app.misc_helpers

    default_params = mw.default_parameters
    default_data = default_params.data if hasattr(default_params, "data") else default_params
    PD = misc.ParametersDict

    # 1. wipe whatever the previous target left behind
    app.card_actions.clear_target_faces(mw, refresh_frame=False)
    vca.remove_all_markers(mw)

    # 2. global settings
    loaded_control = snap.get("control") or {}
    for key in list(mw.control.keys()):
        if key in loaded_control:
            mw.control[key] = copy.deepcopy(loaded_control[key])
    cwa.set_control_widgets_values(mw)
    out_folder = mw.control.get("OutputMediaFolder", "")
    cwa.create_control(mw, "OutputMediaFolder", out_folder)
    try:
        mw.outputFolderLineEdit.setText(out_folder)
    except Exception:
        pass

    # 3. parameters new faces start from
    cur = snap.get("current_widget_parameters")
    if cur:
        mw.current_widget_parameters = PD(
            sla.sanitize_state_dictionary(cur, default_data), default_data
        )

    # 4. target faces
    for tf in snap.get("target_faces", []):
        face_id = tf["face_id"]
        face_img = np.ascontiguousarray(tf["cropped_face"])
        if face_img.ndim == 3 and face_img.shape[2] == 3:
            h, w = face_img.shape[:2]
            q_image = QtGui.QImage(
                face_img.data, w, h, 3 * w, QtGui.QImage.Format_BGR888
            ).copy()
        else:
            q_image = QtGui.QImage(112, 112, QtGui.QImage.Format_RGB888)
            q_image.fill(QtCore.Qt.GlobalColor.gray)
        lva.add_media_thumbnail_to_target_faces_list(
            mw, face_img, _copy_store(tf["embedding_store"]), q_image, face_id
        )
        params = tf.get("parameters")
        if params:
            mw.parameters[face_id] = PD(
                sla.sanitize_state_dictionary(params, default_data), default_data
            )

    def assign_all():
        for tf in snap.get("target_faces", []):
            btn = mw.target_faces.get(tf["face_id"])
            if btn is None:
                continue
            btn.assigned_input_faces.clear()
            btn.assigned_merged_embeddings.clear()
            for input_id, info in tf.get("assigned_input_faces", {}).items():
                btn.assigned_input_faces[input_id] = _copy_store(info["embedding_store"])
            for emb_id, info in tf.get("assigned_merged_embeddings", {}).items():
                btn.assigned_merged_embeddings[emb_id] = _copy_store(info["embedding_store"])
            btn.calculate_assigned_input_embedding()

    def warn_missing_sources():
        merged_live = lva.get_all_merged_embeddings(mw)
        missing = set()
        for tf in snap.get("target_faces", []):
            for input_id, info in tf.get("assigned_input_faces", {}).items():
                if input_id not in mw.input_faces:
                    missing.add(os.path.basename(info.get("media_path") or input_id))
            for emb_id, info in tf.get("assigned_merged_embeddings", {}).items():
                if emb_id not in merged_live:
                    missing.add(str(info.get("name") or emb_id))
        if missing:
            log(
                "Warning: these source faces/embeddings are not loaded in the app any more "
                "(the stored embeddings are used, denoiser K/V references may be unavailable): "
                + ", ".join(sorted(missing))
            )

    assign_all()
    warn_missing_sources()

    # 5. markers
    for pos, mdata in (snap.get("markers") or {}).items():
        single = sla.sanitize_markers_dictionary(mw, {pos: mdata})
        single = sla.convert_markers_to_supported_type(mw, single, PD)
        md = single[pos]
        vca.add_marker(mw, md["parameters"], md["control"], int(pos))

    # 6. segments
    segments = []
    for s in snap.get("segments", []):
        entry = {
            "start": int(s["start"]),
            "end": int(s["end"]),
            "kind": s.get("kind", "normal"),
            "parameters": None,
            "control": copy.deepcopy(s.get("control")),
        }
        raw = s.get("parameters")
        if raw is not None:
            entry["parameters"] = {
                str(fid): PD(dict(pd), default_data) for fid, pd in raw.items()
            }
        segments.append(entry)
    mw.segments = segments
    mw._in_normal_segment = False
    mw._segments_base_parameters = mw.parameters
    mw._segments_base_control = mw.control

    # 7. dropped frames
    vca.set_dropped_frames(mw, snap.get("dropped_frames", []))
    try:
        mw.videoSeekSlider.update()
        vca.update_drop_frame_button_label(mw)
    except Exception:
        pass

    # 8. swap / edit buttons
    mw.swapfacesButton.setChecked(bool(snap.get("swap_faces_enabled", False)))
    mw.editFacesButton.setChecked(bool(snap.get("edit_faces_enabled", False)))
    app.control_actions.handle_face_editor_button_click(mw)

    # 9. select a face (updates all parameter widgets), then re-apply the
    #    assignments because "Keep Input"/"Auto Swap" overwrite them on click.
    wanted = snap.get("selected_target_face_id")
    face_id = wanted if wanted in mw.target_faces else (
        next(iter(mw.target_faces), None)
    )
    if face_id is not None:
        mw.target_faces[face_id].load_target_face()
        assign_all()
        app.card_actions.uncheck_all_input_faces(mw)
        app.card_actions.uncheck_all_merged_embeddings(mw)
        btn = mw.target_faces[face_id]
        merged_live = lva.get_all_merged_embeddings(mw)
        for input_id in btn.assigned_input_faces:
            if input_id in mw.input_faces:
                mw.input_faces[input_id].setChecked(True)
        for emb_id in btn.assigned_merged_embeddings:
            if emb_id in merged_live:
                merged_live[emb_id].setChecked(True)
        mw.current_kv_tensors_map = getattr(btn, "assigned_kv_map", None)
        mw.current_widget_parameters = mw.parameters[face_id].copy()
    else:
        cwa.set_widgets_values_using_face_id_parameters(mw, face_id=None)

    try:
        mw.video_processor.ui_state_is_dirty = True
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Persistent queue
# --------------------------------------------------------------------------- #
class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.jobs_dir = root / "jobs"
        self.index_path = root / "queue.json"
        self.log_path = root / "job_manager.log"
        self.jobs: list[dict] = []
        self.settings: dict = dict(DEFAULT_SETTINGS)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ---- index ---------------------------------------------------------
    def load(self):
        self.jobs = []
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            backup = self.index_path.with_suffix(".json.damaged")
            try:
                self.index_path.replace(backup)
            except OSError:
                pass
            print(f"[JOB MANAGER] queue.json unreadable ({exc}); moved to {backup.name}")
            return
        saved = data.get("settings", {})
        for key, default in DEFAULT_SETTINGS.items():
            self.settings[key] = saved.get(key, default)
        for job in data.get("jobs", []):
            if not isinstance(job, dict) or "id" not in job:
                continue
            if not self.snapshot_path(job["id"]).is_file():
                continue
            if job.get("status") == ST_RUNNING:
                job["status"] = ST_PENDING
                job["message"] = "Interrupted (the app closed while this job was running)."
            self.jobs.append(job)

    def save(self):
        payload = json.dumps(
            {"version": 1, "settings": self.settings, "jobs": self.jobs},
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        _atomic_write_bytes(self.index_path, payload)

    # ---- snapshots -----------------------------------------------------
    def snapshot_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.pkl"

    def write_snapshot(self, job_id: str, snap: dict):
        _atomic_write_bytes(
            self.snapshot_path(job_id),
            pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def read_snapshot(self, job_id: str) -> dict:
        path = self.snapshot_path(job_id)
        if not path.is_file():
            raise JobError("The stored job data is missing.")
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def delete_snapshot(self, job_id: str):
        try:
            self.snapshot_path(job_id).unlink()
        except OSError:
            pass

    # ---- jobs ----------------------------------------------------------
    def get(self, job_id: str):
        for job in self.jobs:
            if job["id"] == job_id:
                return job
        return None

    def new_job_meta(self, snap: dict, name: str, output_name: str = "") -> dict:
        return {
            "id": uuid.uuid4().hex,
            "name": name,
            "output_name": output_name,
            "media_path": snap["media_path"],
            "file_type": snap["file_type"],
            "enabled": True,
            "status": ST_PENDING,
            "message": "",
            "created": _now_str(),
            "started": "",
            "finished": "",
            "output_path": "",
            "attempts": 0,
            **self.describe(snap),
        }

    @staticmethod
    def describe(snap: dict) -> dict:
        return {
            "n_faces": len(snap.get("target_faces", [])),
            "n_markers": len(snap.get("markers", {})),
            "n_segments": len(snap.get("segments", [])),
        }

    def log(self, text: str):
        line = f"{_now_str()}  {text}"
        print(f"[JOB MANAGER] {text}")
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
        return line


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class JobRunner(QtCore.QObject):
    log_line = QtCore.Signal(str)
    job_changed = QtCore.Signal(str)
    progress = QtCore.Signal(int, int, str)
    state_changed = QtCore.Signal(bool)
    queue_finished = QtCore.Signal(str)

    def __init__(self, mw, store: JobStore):
        super().__init__(mw)
        self.mw = mw
        self.store = store
        self.active = False
        self._queue: list[str] = []
        self._job = None
        self._snap = None
        self._phase = None  # "loading" | "running"
        self._stop_after = False
        self._abort = False
        self._timed_out = False
        self._started_seen = False
        self._t_job = 0.0
        self._t_phase = 0.0
        self._out_folder = ""
        self._image_output = ""
        self._pre_outputs: dict = {}
        self._counts = {"done": 0, "failed": 0}
        self._awake_holder: dict = {}
        self._loader = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    # ---- public --------------------------------------------------------
    def say(self, text: str):
        self.log_line.emit(self.store.log(text))

    def start(self, job_ids):
        if self.active:
            return False
        ids = [j for j in job_ids if self.store.get(j) is not None]
        if not ids:
            self.say("Nothing to run.")
            return False
        vp = self.mw.video_processor
        app = _app()
        if app.video_control_actions.is_issue_scan_active(self.mw):
            self.say("Cannot start: an issue scan is running.")
            return False
        if vp.processing or vp.is_processing_segments or getattr(vp, "finalization_in_progress", False):
            self.say("Cannot start: the app is currently playing or recording.")
            return False
        self._queue = list(ids)
        self._stop_after = False
        self._abort = False
        self._counts = {"done": 0, "failed": 0}
        self.active = True
        if self.store.settings.get("keep_awake"):
            _set_keep_awake(True, self._awake_holder)
        self.state_changed.emit(True)
        self.say(f"=== Queue started: {len(ids)} job(s) ===")
        self._timer.start()
        QtCore.QTimer.singleShot(0, self._advance)
        return True

    def stop_after_current(self):
        if self.active:
            self._stop_after = True
            self.say("Will stop after the current job.")

    def abort(self):
        if not self.active:
            return
        self._abort = True
        self.say("Abort requested.")
        try:
            if self._phase == "running":
                self._started_seen = True
                self.mw.video_processor.stop_processing()
        except Exception as exc:
            self.say(f"Abort: stop_processing failed: {exc}")

    def shutdown(self):
        """App is closing."""
        self._timer.stop()
        if self._job is not None and self._job.get("status") == ST_RUNNING:
            self._job["status"] = ST_PENDING
            self._job["message"] = "Interrupted (the app closed while this job was running)."
        try:
            self.store.save()
        except Exception:
            pass
        if self.active:
            _set_keep_awake(False, self._awake_holder)
        self.active = False

    # ---- queue flow ----------------------------------------------------
    def _advance(self):
        if not self.active:
            return
        if self._abort:
            return self._finish_queue("aborted")
        if self._stop_after:
            return self._finish_queue("stopped")
        if not self._queue:
            return self._finish_queue("completed")
        job = self.store.get(self._queue.pop(0))
        if job is None:
            return self._advance()
        self._begin(job)

    def _begin(self, job):
        self._job = job
        self._phase = None
        self._timed_out = False
        self._started_seen = False
        self._image_output = ""
        self._out_folder = ""
        self._pre_outputs = {}
        job["status"] = ST_RUNNING
        job["message"] = ""
        job["started"] = _now_str()
        job["finished"] = ""
        job["output_path"] = ""
        self._t_job = time.time()
        self._save_and_notify(job)
        self.say(f"--- Starting job '{job['name']}' ({os.path.basename(job['media_path'])}) ---")
        try:
            self._snap = self.store.read_snapshot(job["id"])
            self._check_job_preconditions(job, self._snap)
            card = self._find_card(job["media_path"])
            if card is not None:
                QtCore.QTimer.singleShot(0, self._restore_and_start)
            else:
                self._start_target_loader(job)
        except JobError as exc:
            self._job_failed(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._job_failed(f"Unexpected error while preparing the job: {exc}")

    def _check_job_preconditions(self, job, snap):
        if not os.path.isfile(job["media_path"]):
            raise JobError(f"Target file not found: {job['media_path']}")
        out = str((snap.get("control") or {}).get("OutputMediaFolder", "")).strip()
        if not out:
            raise JobError("The job has no output folder set.")
        if snap.get("file_type") == "video" and not _app().misc_helpers.is_ffmpeg_in_path():
            raise JobError("FFmpeg was not found in the PATH.")

    # ---- target media --------------------------------------------------
    def _find_card(self, path):
        want = _norm(path)
        for card in self.mw.target_videos.values():
            try:
                if _norm(card.media_path) == want:
                    return card
            except RuntimeError:
                continue
        return None

    def _start_target_loader(self, job):
        app = _app()
        self.say("Target is not in the target list, adding it...")
        from functools import partial

        worker = app.ui_workers.TargetMediaLoaderWorker(
            main_window=self.mw,
            folder_name=False,
            files_list=[job["media_path"]],
            media_ids=[str(uuid.uuid1().int)],
            sort_files_list_by_name=False,
            metadata_enabled=False,
        )
        worker.thumbnail_ready.connect(
            partial(app.list_view_actions.add_media_thumbnail_to_target_videos_list, self.mw),
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self.mw.video_loader_worker = worker
        self._loader = worker
        self._phase = "loading"
        self._t_phase = time.time()
        worker.start()

    # ---- restore + start -------------------------------------------------
    def _restore_and_start(self):
        job, snap = self._job, self._snap
        if not self.active or job is None:
            return
        try:
            mw = self.mw
            vp = mw.video_processor
            if vp.processing or vp.is_processing_segments:
                vp.stop_processing()

            card = self._find_card(job["media_path"])
            if card is None:
                raise JobError("Could not add the target media to the target list.")
            card.setChecked(True)
            card.load_media()
            if mw.selected_video_button is not card or _norm(vp.media_path or "") != _norm(card.media_path):
                raise JobError("The target media could not be opened.")

            restore_snapshot(mw, snap, log=self.say)
            self._start_processing(job, snap)
        except JobError as exc:
            self._job_failed(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._job_failed(f"Could not start the job: {exc}")

    def _start_processing(self, job, snap):
        app = _app()
        mw = self.mw
        vp = mw.video_processor
        vca = app.video_control_actions
        out_base = str(mw.control.get("OutputMediaFolder", "")).strip()
        out_folder = vca.resolve_output_folder(mw, str(job["media_path"]))
        os.makedirs(out_folder or out_base, exist_ok=True)
        self._out_folder = out_folder
        self._pre_outputs = self._scan_outputs(out_folder)

        mw.current_job_name = job["name"]
        mw.use_job_name_for_output = False
        mw.output_file_name = (job.get("output_name") or "").strip() or None

        self._t_phase = time.time()
        self._t_job = time.time()
        self._started_seen = False

        if snap.get("file_type") == "image":
            self._process_image(job, out_folder)
            return

        try:
            vp.sequential_detector.reset_state()
        except Exception:
            pass
        try:
            vp.stopped_by_error_limit = False
        except Exception:
            pass

        segments = [
            (int(s["start"]), int(s["end"]))
            for s in getattr(mw, "segments", [])
            if s.get("kind") == "recording" and s["end"] > s["start"]
        ]
        segments = [(a, b) for a, b in vca.apply_exclude_segments(mw, segments) if a < b]

        if vp.max_frame_number is None or vp.max_frame_number <= 0:
            raise JobError("Cannot determine the video length.")

        vca._disable_compare_preview_modes_for_recording(mw)
        mw.buttonMediaRecord.blockSignals(True)
        mw.buttonMediaRecord.setChecked(True)
        mw.buttonMediaRecord.blockSignals(False)
        vca.set_record_button_icon_to_stop(mw)
        mw.buttonMediaPlay.setEnabled(False)

        self._phase = "running"
        if segments:
            self.say(f"Recording {len(segments)} segment(s)...")
            vp.start_multi_segment_recording(segments, triggered_by_job_manager=True)
        else:
            self.say("Recording the full video...")
            mw.videoSeekSlider.blockSignals(True)
            mw.videoSeekSlider.setValue(0)
            mw.videoSeekSlider.blockSignals(False)
            vp.current_frame_number = 0
            mw.job_manager_initiated_record = True
            vp.recording = True
            vp.process_video()
        folder = getattr(vp, "active_output_folder", "")
        if folder and _norm(folder) != _norm(out_folder):
            self._out_folder = folder
            self._pre_outputs.update(self._scan_outputs(folder))

    def _process_image(self, job, out_folder):
        app = _app()
        mw = self.mw
        vp = mw.video_processor
        from PIL import Image

        self._phase = None
        vp.process_current_frame(synchronous=True)
        frame = vp.current_frame
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise JobError("Processing returned an empty frame.")
        use_jpeg = bool(mw.control.get("ImageFormatToggle", False))
        fmt = "jpegimage" if use_jpeg else "image"
        name = (job.get("output_name") or "").strip()
        stamp = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
        os.makedirs(out_folder, exist_ok=True)
        path = app.misc_helpers.get_output_file_path(
            job["media_path"],
            out_folder,
            media_type=fmt,
            output_file_name=f"{name}_{stamp}" if name else None,
        )
        pil = Image.fromarray(frame[..., ::-1])
        if use_jpeg:
            pil.save(path, "JPEG", quality=95)
        else:
            pil.save(path, "PNG")
        self._image_output = path
        self._started_seen = True
        QtCore.QTimer.singleShot(0, self._finish_job)

    # ---- polling -----------------------------------------------------------
    def _tick(self):
        if not self.active:
            self._timer.stop()
            return
        try:
            if self.store.settings.get("dismiss_popups"):
                self._dismiss_popups()
            if self._phase == "loading":
                self._poll_loading()
            elif self._phase == "running":
                self._poll_running()
        except Exception as exc:
            traceback.print_exc()
            self._job_failed(f"Runner error: {exc}")

    def _dismiss_popups(self):
        box = QtWidgets.QApplication.activeModalWidget()
        if not isinstance(box, QtWidgets.QMessageBox):
            return
        text = " ".join(f"{box.windowTitle()}: {box.text()}".split())
        button = box.defaultButton() or box.escapeButton()
        if button is None and box.buttons():
            button = box.buttons()[0]
        self.say(f"Auto-dismissed a pop-up while unattended -> {text[:300]}")
        if button is not None:
            button.click()
        else:
            box.close()

    def _poll_loading(self):
        worker = self._loader
        app = _app()
        busy = worker is not None and worker.isRunning()
        pending = app.list_view_actions._has_pending_target_media_thumbnail_work(self.mw)
        if busy or pending:
            if time.time() - self._t_phase > 120:
                self._job_failed("Timed out while adding the target to the target list.")
            return
        if self._find_card(self._job["media_path"]) is None:
            self._job_failed("The target media could not be added to the target list.")
            return
        self._phase = None
        QtCore.QTimer.singleShot(0, self._restore_and_start)

    def _poll_running(self):
        vp = self.mw.video_processor
        active = bool(
            vp.processing
            or vp.is_processing_segments
            or vp.recording
            or getattr(vp, "finalization_in_progress", False)
        )
        if active:
            self._started_seen = True
            try:
                cur = int(max(vp.current_frame_number, getattr(vp, "next_frame_to_display", 0) or 0))
                total = int(vp.max_frame_number or 0)
                self.progress.emit(cur, total, self._job["name"])
            except Exception:
                pass
            limit = float(self.store.settings.get("max_minutes", 0) or 0)
            if limit > 0 and not self._timed_out and time.time() - self._t_job > limit * 60:
                self._timed_out = True
                self.say(f"Job exceeded the time limit of {limit} min, stopping it.")
                vp.stop_processing()
            return
        if not self._started_seen and not self._abort:
            if time.time() - self._t_phase > 20:
                self._job_failed("Recording did not start.")
            return
        self._phase = None
        QtCore.QTimer.singleShot(0, self._finish_job)

    # ---- job end ---------------------------------------------------------
    @staticmethod
    def _scan_outputs(folder):
        """{path: (mtime, size)} of the media files that exist in a folder right now."""
        found = {}
        if not folder or not os.path.isdir(folder):
            return found
        for entry in os.scandir(folder):
            try:
                if entry.is_file() and entry.name.lower().endswith(OUTPUT_SUFFIXES):
                    st = entry.stat()
                    found[entry.path] = (st.st_mtime, st.st_size)
            except OSError:
                continue
        return found

    def _find_output(self):
        """Newest output file that did not exist (or changed) since the job started."""
        if self._image_output and os.path.isfile(self._image_output):
            return self._image_output
        current = self._scan_outputs(self._out_folder)
        fresh = [
            (stat[0], path)
            for path, stat in current.items()
            if self._pre_outputs.get(path) != stat and stat[1] > 0
        ]
        return max(fresh)[1] if fresh else ""

    def _finish_job(self):
        job = self._job
        if job is None or not self.active:
            return
        vp = self.mw.video_processor
        if self._abort:
            job["status"] = ST_PENDING
            job["message"] = "Aborted by user."
            job["finished"] = _now_str()
            self._save_and_notify(job)
            self.say(f"Job '{job['name']}' aborted.")
            self._job = None
            return self._finish_queue("aborted")
        if self._timed_out:
            return self._job_failed("Stopped: exceeded the per-job time limit.")
        if bool(getattr(vp, "stopped_by_error_limit", False)):
            return self._job_failed(
                "Stopped by the read-error limit; the output (if any) is incomplete."
            )
        output = self._find_output()
        if job["file_type"] == "video" and not output:
            return self._job_failed(f"Finished, but no output file was found in: {self._out_folder}")
        job["status"] = ST_DONE
        job["message"] = ""
        job["output_path"] = output
        job["finished"] = _now_str()
        job["attempts"] = 0
        self._counts["done"] += 1
        self._save_and_notify(job)
        elapsed = int(time.time() - self._t_job)
        self.say(
            f"Job '{job['name']}' finished in {elapsed // 3600:d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}"
            + (f" -> {output}" if output else "")
        )
        self._job = None
        self._after_job()

    def _job_failed(self, message: str):
        job = self._job
        self._phase = None
        if job is None:
            return
        vp = self.mw.video_processor
        try:
            if vp.processing or vp.is_processing_segments or vp.recording:
                vp.stop_processing()
        except Exception:
            pass
        self.say(f"Job '{job['name']}' failed: {message}")
        retries = int(self.store.settings.get("retries", 0) or 0)
        if job.get("attempts", 0) < retries and not self._abort and not self._stop_after:
            job["attempts"] = job.get("attempts", 0) + 1
            job["status"] = ST_PENDING
            job["message"] = f"Retrying ({job['attempts']}/{retries}) after: {message}"
            self._save_and_notify(job)
            self._queue.insert(0, job["id"])
            self._job = None
            self._after_job()
            return
        job["status"] = ST_FAILED
        job["message"] = message
        job["finished"] = _now_str()
        self._counts["failed"] += 1
        self._save_and_notify(job)
        self._job = None
        if not self.store.settings.get("continue_on_error", True):
            self.say("Stopping the queue because 'continue after a failed job' is off.")
            self._finish_queue("error")
            return
        self._after_job()

    def _after_job(self):
        gc.collect()
        QtCore.QTimer.singleShot(1500, self._advance)

    def _save_and_notify(self, job):
        try:
            self.store.save()
        except Exception as exc:
            print(f"[JOB MANAGER] Could not save queue: {exc}")
        self.job_changed.emit(job["id"])

    def _finish_queue(self, reason: str):
        self._timer.stop()
        self.active = False
        self._phase = None
        _set_keep_awake(False, self._awake_holder)
        summary = (
            f"Queue {reason}: {self._counts['done']} done, {self._counts['failed']} failed"
            + (f", {len(self._queue)} not run" if self._queue else "")
        )
        self.say(f"=== {summary} ===")
        self._queue = []
        self.state_changed.emit(False)
        self.queue_finished.emit(summary)
        try:
            self.store.save()
        except Exception:
            pass
        if reason == "completed":
            self._run_after_action(self.store.settings.get("after_action", "none"))

    def _run_after_action(self, action: str):
        if action in (None, "", "none"):
            return
        self.say(f"Running 'when finished' action: {action}")
        try:
            if action == "close":
                QtCore.QTimer.singleShot(2000, self.mw.close)
                return
            if sys.platform.startswith("win"):
                cmds = {
                    "sleep": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                    "hibernate": ["shutdown", "/h"],
                    "shutdown": ["shutdown", "/s", "/t", "60", "/c", "VisoMaster Job Manager finished"],
                }
            elif sys.platform == "darwin":
                cmds = {
                    "sleep": ["pmset", "sleepnow"],
                    "hibernate": ["pmset", "sleepnow"],
                    "shutdown": ["osascript", "-e", 'tell app "System Events" to shut down'],
                }
            else:
                cmds = {
                    "sleep": ["systemctl", "suspend"],
                    "hibernate": ["systemctl", "hibernate"],
                    "shutdown": ["systemctl", "poweroff"],
                }
            cmd = cmds.get(action)
            if cmd is None:
                return
            # Let the app save its workspace first for a shutdown.
            if action == "shutdown":
                subprocess.Popen(cmd)
                QtCore.QTimer.singleShot(3000, self.mw.close)
            else:
                QtCore.QTimer.singleShot(2000, lambda c=cmd: subprocess.Popen(c))
        except Exception as exc:
            self.say(f"'When finished' action failed: {exc}")


# --------------------------------------------------------------------------- #
# UI + extension entry
# --------------------------------------------------------------------------- #
class JobManagerExtension:
    def __init__(self, main_window):
        self.mw = main_window
        root = Path(main_window.project_root_path) / DATA_DIR_NAME
        self.store = JobStore(root)
        self.store.load()
        self.runner = JobRunner(main_window, self.store)
        self._refreshing = False
        self._build_ui()
        self.runner.log_line.connect(self._append_log)
        self.runner.job_changed.connect(self._on_job_changed)
        self.runner.progress.connect(self._on_progress)
        self.runner.state_changed.connect(self._on_state_changed)
        self.runner.queue_finished.connect(self._on_queue_finished)
        self._refresh_list()
        self._on_state_changed(False)
        self._maybe_autostart()

    # ---- extension hooks -------------------------------------------------
    def on_app_close(self):
        try:
            self.runner.shutdown()
        except Exception:
            traceback.print_exc()

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        lay = QtWidgets.QVBoxLayout(body)

        intro = QtWidgets.QLabel(
            "Set up a target the way you want it, then add it as a job. "
            "Run all jobs later as one unattended batch."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # --- queue -----------------------------------------------------------
        self.add_btn = QtWidgets.QPushButton("Add current target as job")
        self.add_btn.setToolTip(
            "Stores the selected target with its faces, source assignments, parameters, "
            "markers, segments and all global settings."
        )
        self.add_btn.clicked.connect(self._add_job)
        lay.addWidget(self.add_btn)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setMinimumHeight(170)
        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(lambda *_: self._load_into_editor())
        lay.addWidget(self.list)
        hint = QtWidgets.QLabel("Tick = job takes part in 'Run all'. Double-click = load it into the app.")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        grid = QtWidgets.QGridLayout()
        self.btn_up = self._button("Move up", lambda: self._move(-1))
        self.btn_down = self._button("Move down", lambda: self._move(1))
        self.btn_update = self._button("Update from app", self._update_job,
                                       "Replace the selected job with the current state of the app.")
        self.btn_load = self._button("Load into app", self._load_into_editor,
                                     "Restore the selected job in the main window so you can review or edit it.")
        self.btn_dup = self._button("Duplicate", self._duplicate)
        self.btn_remove = self._button("Remove", self._remove)
        self.btn_requeue = self._button("Set to pending", self._requeue)
        self.btn_open = self._button("Open output folder", self._open_output)
        buttons = [self.btn_up, self.btn_down, self.btn_update, self.btn_load,
                   self.btn_dup, self.btn_remove, self.btn_requeue, self.btn_open]
        for i, b in enumerate(buttons):
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)

        # --- selected job ------------------------------------------------------
        box = QtWidgets.QGroupBox("Selected job")
        form = QtWidgets.QFormLayout(box)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(self._commit_details)
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("blank = <target name>_<timestamp>")
        self.output_edit.setToolTip("Output file name (without extension). A timestamp is appended.")
        self.output_edit.editingFinished.connect(self._commit_details)
        form.addRow("Name", self.name_edit)
        form.addRow("Output name", self.output_edit)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.info_label)
        lay.addWidget(box)

        # --- run -----------------------------------------------------------------
        run_box = QtWidgets.QGroupBox("Run")
        run_lay = QtWidgets.QVBoxLayout(run_box)
        self.run_all_btn = QtWidgets.QPushButton("Run all ticked pending jobs")
        self.run_all_btn.clicked.connect(self._run_all)
        self.run_sel_btn = QtWidgets.QPushButton("Run selected job(s)")
        self.run_sel_btn.clicked.connect(self._run_selected)
        self.stop_after_btn = QtWidgets.QPushButton("Stop after current job")
        self.stop_after_btn.clicked.connect(lambda: self.runner.stop_after_current())
        self.abort_btn = QtWidgets.QPushButton("Abort now")
        self.abort_btn.clicked.connect(self._confirm_abort)
        for b in (self.run_all_btn, self.run_sel_btn, self.stop_after_btn, self.abort_btn):
            run_lay.addWidget(b)
        self.progress_label = QtWidgets.QLabel("Idle")
        self.progress_label.setWordWrap(True)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        run_lay.addWidget(self.progress_label)
        run_lay.addWidget(self.progress_bar)
        lay.addWidget(run_box)

        # --- options ---------------------------------------------------------------
        opt_box = QtWidgets.QGroupBox("Unattended run options")
        opt = QtWidgets.QFormLayout(opt_box)
        s = self.store.settings
        self.cb_continue = QtWidgets.QCheckBox("Continue with the next job after a failure")
        self.cb_continue.setChecked(bool(s["continue_on_error"]))
        self.cb_awake = QtWidgets.QCheckBox("Keep the PC awake while jobs run")
        self.cb_awake.setChecked(bool(s["keep_awake"]))
        self.cb_popups = QtWidgets.QCheckBox("Auto-dismiss pop-up dialogs while jobs run")
        self.cb_popups.setToolTip("Message boxes are closed with their default button and written to the log.")
        self.cb_popups.setChecked(bool(s["dismiss_popups"]))
        self.cb_autostart = QtWidgets.QCheckBox("Start pending jobs automatically when the app starts")
        self.cb_autostart.setToolTip(
            "Same as launching the app with --run-jobs. Use it with a scheduled task for overnight runs."
        )
        self.cb_autostart.setChecked(bool(s["auto_start_on_launch"]))
        for cb in (self.cb_continue, self.cb_awake, self.cb_popups, self.cb_autostart):
            cb.toggled.connect(self._commit_options)
            opt.addRow(cb)
        self.spin_retries = QtWidgets.QSpinBox()
        self.spin_retries.setRange(0, 5)
        self.spin_retries.setValue(int(s["retries"]))
        self.spin_retries.valueChanged.connect(self._commit_options)
        self.spin_minutes = QtWidgets.QSpinBox()
        self.spin_minutes.setRange(0, 24 * 60)
        self.spin_minutes.setSpecialValueText("no limit")
        self.spin_minutes.setSuffix(" min")
        self.spin_minutes.setValue(int(s["max_minutes"]))
        self.spin_minutes.valueChanged.connect(self._commit_options)
        self.combo_after = QtWidgets.QComboBox()
        for key, label in AFTER_ACTIONS:
            self.combo_after.addItem(label, key)
        idx = max(0, self.combo_after.findData(s["after_action"]))
        self.combo_after.setCurrentIndex(idx)
        self.combo_after.currentIndexChanged.connect(self._commit_options)
        opt.addRow("Retries per job", self.spin_retries)
        opt.addRow("Time limit per job", self.spin_minutes)
        opt.addRow("When finished", self.combo_after)
        lay.addWidget(opt_box)

        # --- log -----------------------------------------------------------------
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMinimumHeight(140)
        lay.addWidget(QtWidgets.QLabel("Log"))
        lay.addWidget(self.log_view)
        lay.addStretch()

        self.tab = outer
        add_tab = getattr(self.mw, "add_parameter_tab", None)
        if add_tab is not None:
            add_tab(outer, "Jobs")
        else:
            self.mw.tabWidget.addTab(outer, "Jobs")

    @staticmethod
    def _button(text, slot, tip=""):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(slot)
        if tip:
            b.setToolTip(tip)
        return b

    # ---- list handling -----------------------------------------------------
    def _selected_ids(self):
        return [it.data(QtCore.Qt.ItemDataRole.UserRole) for it in self.list.selectedItems()]

    def _item_text(self, index, job):
        return (
            f"{index + 1}. {job['name']}  -  {os.path.basename(job['media_path'])}"
            f"  [{STATUS_LABEL.get(job['status'], job['status'])}]"
        )

    def _style_item(self, item, job):
        color = None
        if job["status"] == ST_DONE:
            color = QtGui.QColor(70, 160, 90)
        elif job["status"] == ST_FAILED:
            color = QtGui.QColor(210, 70, 70)
        elif job["status"] == ST_RUNNING:
            color = QtGui.QColor(230, 160, 40)
        elif not job.get("enabled", True):
            color = QtGui.QColor(140, 140, 140)
        item.setForeground(QtGui.QBrush(color) if color else QtGui.QBrush())
        item.setToolTip(
            f"{job['media_path']}\nFaces: {job.get('n_faces', 0)}   Markers: {job.get('n_markers', 0)}   "
            f"Segments: {job.get('n_segments', 0)}\n{job.get('message', '')}".strip()
        )

    def _refresh_list(self, keep_ids=None):
        keep = set(keep_ids if keep_ids is not None else self._selected_ids())
        self._refreshing = True
        try:
            self.list.clear()
            for i, job in enumerate(self.store.jobs):
                item = QtWidgets.QListWidgetItem(self._item_text(i, job))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, job["id"])
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if job.get("enabled", True) else QtCore.Qt.CheckState.Unchecked
                )
                self._style_item(item, job)
                self.list.addItem(item)
                if job["id"] in keep:
                    item.setSelected(True)
        finally:
            self._refreshing = False
        self._on_selection_changed()

    def _on_job_changed(self, job_id):
        keep = self._selected_ids()
        self._refresh_list(keep)

    def _on_item_changed(self, item):
        if self._refreshing:
            return
        job = self.store.get(item.data(QtCore.Qt.ItemDataRole.UserRole))
        if job is None:
            return
        job["enabled"] = item.checkState() == QtCore.Qt.CheckState.Checked
        self.store.save()
        self._style_item(item, job)

    def _on_selection_changed(self):
        ids = self._selected_ids()
        single = len(ids) == 1
        job = self.store.get(ids[0]) if single else None
        for w in (self.name_edit, self.output_edit):
            w.setEnabled(single and not self.runner.active)
        self.name_edit.setText(job["name"] if job else "")
        self.output_edit.setText(job.get("output_name", "") if job else "")
        if job:
            lines = [
                f"Target: {job['media_path']}",
                f"Faces: {job.get('n_faces', 0)}   Markers: {job.get('n_markers', 0)}   "
                f"Segments: {job.get('n_segments', 0)}",
                f"Status: {STATUS_LABEL.get(job['status'], job['status'])}",
            ]
            if job.get("message"):
                lines.append(job["message"])
            if job.get("output_path"):
                lines.append(f"Output: {job['output_path']}")
            if job.get("finished"):
                lines.append(f"Finished: {job['finished']}")
            self.info_label.setText("\n".join(lines))
        else:
            self.info_label.setText("Select one job to see its details." if not ids else f"{len(ids)} jobs selected.")

    def _commit_details(self):
        ids = self._selected_ids()
        if len(ids) != 1:
            return
        job = self.store.get(ids[0])
        if job is None:
            return
        new_name = self.name_edit.text().strip() or job["name"]
        job["name"] = new_name
        job["output_name"] = self.output_edit.text().strip()
        self.store.save()
        self._refresh_list([job["id"]])

    def _commit_options(self, *_):
        s = self.store.settings
        s["continue_on_error"] = self.cb_continue.isChecked()
        s["keep_awake"] = self.cb_awake.isChecked()
        s["dismiss_popups"] = self.cb_popups.isChecked()
        s["auto_start_on_launch"] = self.cb_autostart.isChecked()
        s["retries"] = self.spin_retries.value()
        s["max_minutes"] = self.spin_minutes.value()
        s["after_action"] = self.combo_after.currentData()
        try:
            self.store.save()
        except Exception as exc:
            print(f"[JOB MANAGER] Could not save options: {exc}")

    # ---- actions ---------------------------------------------------------------
    def _info(self, title, text):
        QtWidgets.QMessageBox.information(self.mw, title, text)

    def _busy_message(self):
        vp = self.mw.video_processor
        if self.runner.active:
            self._info("Job Manager", "The queue is running.")
            return True
        if vp.processing or vp.is_processing_segments or getattr(vp, "finalization_in_progress", False):
            self._info("Job Manager", "Stop playback / recording first.")
            return True
        return False

    def _capture_checked(self):
        """Capture the current state, running the sanity checks. Returns snapshot or None."""
        if self._busy_message():
            return None
        try:
            snap = capture_snapshot(self.mw)
        except JobError as exc:
            self._info("Job Manager", str(exc))
            return None
        except Exception as exc:
            traceback.print_exc()
            self._info("Job Manager", f"Could not read the current state:\n{exc}")
            return None
        if not str(snap["control"].get("OutputMediaFolder", "")).strip():
            self._info("Job Manager", "Select an output folder first.")
            return None
        if not snap["target_faces"]:
            answer = QtWidgets.QMessageBox.question(
                self.mw,
                "No target faces",
                "No target faces are defined, so the output would not be changed.\n\nAdd the job anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return None
        return snap

    def _add_job(self):
        snap = self._capture_checked()
        if snap is None:
            return
        job = self.store.new_job_meta(snap, Path(snap["media_path"]).stem)
        try:
            self.store.write_snapshot(job["id"], snap)
        except Exception as exc:
            traceback.print_exc()
            self._info("Job Manager", f"Could not store the job:\n{exc}")
            return
        self.store.jobs.append(job)
        self.store.save()
        self._refresh_list([job["id"]])
        self.runner.say(f"Job added: '{job['name']}' ({job['n_faces']} face(s), "
                        f"{job['n_markers']} marker(s), {job['n_segments']} segment(s)).")

    def _single_selected(self):
        ids = self._selected_ids()
        if len(ids) != 1:
            self._info("Job Manager", "Select exactly one job.")
            return None
        return self.store.get(ids[0])

    def _update_job(self):
        job = self._single_selected()
        if job is None:
            return
        snap = self._capture_checked()
        if snap is None:
            return
        self.store.write_snapshot(job["id"], snap)
        job.update(
            media_path=snap["media_path"],
            file_type=snap["file_type"],
            status=ST_PENDING,
            message="Updated from the app.",
            output_path="",
            **self.store.describe(snap),
        )
        self.store.save()
        self._refresh_list([job["id"]])
        self.runner.say(f"Job '{job['name']}' updated from the current state.")

    def _load_into_editor(self):
        if self._busy_message():
            return
        job = self._single_selected()
        if job is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self.mw,
            "Load job",
            "This replaces the current target faces, markers, segments and settings in the app "
            "with the ones stored in the job.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            snap = self.store.read_snapshot(job["id"])
            if not os.path.isfile(job["media_path"]):
                raise JobError(f"Target file not found: {job['media_path']}")
            card = self.runner._find_card(job["media_path"])
            if card is None:
                raise JobError("The target is not in the target list. Add it to the list first.")
            card.setChecked(True)
            card.load_media()
            restore_snapshot(self.mw, snap, log=self.runner.say)
            _app().common_widget_actions.refresh_frame(self.mw)
            self.runner.say(f"Job '{job['name']}' loaded into the app.")
        except JobError as exc:
            self._info("Job Manager", str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._info("Job Manager", f"Could not load the job:\n{exc}")

    def _move(self, delta):
        if self.runner.active:
            return
        ids = self._selected_ids()
        if len(ids) != 1:
            return
        jobs = self.store.jobs
        i = next((k for k, j in enumerate(jobs) if j["id"] == ids[0]), None)
        if i is None:
            return
        j = i + delta
        if 0 <= j < len(jobs):
            jobs[i], jobs[j] = jobs[j], jobs[i]
            self.store.save()
            self._refresh_list([ids[0]])

    def _duplicate(self):
        job = self._single_selected()
        if job is None:
            return
        try:
            snap = self.store.read_snapshot(job["id"])
        except JobError as exc:
            self._info("Job Manager", str(exc))
            return
        clone = copy.deepcopy(job)
        clone.update(id=uuid.uuid4().hex, name=f"{job['name']} (copy)", status=ST_PENDING,
                     message="", output_path="", finished="", started="", created=_now_str(), attempts=0)
        self.store.write_snapshot(clone["id"], snap)
        idx = self.store.jobs.index(job)
        self.store.jobs.insert(idx + 1, clone)
        self.store.save()
        self._refresh_list([clone["id"]])

    def _remove(self):
        if self.runner.active:
            self._info("Job Manager", "The queue is running.")
            return
        ids = self._selected_ids()
        if not ids:
            return
        answer = QtWidgets.QMessageBox.question(
            self.mw, "Remove jobs", f"Remove {len(ids)} job(s)?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for job_id in ids:
            self.store.delete_snapshot(job_id)
        self.store.jobs = [j for j in self.store.jobs if j["id"] not in ids]
        self.store.save()
        self._refresh_list([])

    def _requeue(self):
        if self.runner.active:
            return
        for job_id in self._selected_ids():
            job = self.store.get(job_id)
            if job:
                job.update(status=ST_PENDING, message="", attempts=0)
        self.store.save()
        self._refresh_list()

    def _open_output(self):
        job = self._single_selected()
        if job is None:
            return
        target = job.get("output_path")
        folder = os.path.dirname(target) if target else ""
        if not folder:
            try:
                snap = self.store.read_snapshot(job["id"])
                folder = str(snap["control"].get("OutputMediaFolder", ""))
            except Exception:
                folder = ""
        if folder and os.path.isdir(folder):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))
        else:
            self._info("Job Manager", "The output folder does not exist yet.")

    def _run_all(self):
        ids = [j["id"] for j in self.store.jobs if j.get("enabled", True) and j["status"] == ST_PENDING]
        if not ids:
            self._info("Job Manager", "There are no ticked pending jobs.\nUse 'Set to pending' to queue finished jobs again.")
            return
        self._commit_options()
        self.runner.start(ids)

    def _run_selected(self):
        ids = self._selected_ids()
        if not ids:
            self._info("Job Manager", "Select one or more jobs first.")
            return
        order = {j["id"]: i for i, j in enumerate(self.store.jobs)}
        self._commit_options()
        self.runner.start(sorted(ids, key=lambda x: order.get(x, 0)))

    def _confirm_abort(self):
        answer = QtWidgets.QMessageBox.question(
            self.mw, "Abort", "Stop the current job right now? Its output may be incomplete.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.runner.abort()

    # ---- runner feedback -----------------------------------------------------
    def _append_log(self, line: str):
        self.log_view.appendPlainText(line)

    def _on_progress(self, cur, total, name):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(cur, max(1, total)))
        self.progress_label.setText(f"{name}: frame {cur} / {total}")

    def _on_state_changed(self, running: bool):
        for w in (self.add_btn, self.btn_up, self.btn_down, self.btn_update, self.btn_load,
                  self.btn_dup, self.btn_remove, self.btn_requeue, self.run_all_btn, self.run_sel_btn,
                  self.name_edit, self.output_edit):
            w.setEnabled(not running)
        self.stop_after_btn.setEnabled(running)
        self.abort_btn.setEnabled(running)
        if running:
            self.progress_label.setText("Starting...")
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
        self._on_selection_changed()

    def _on_queue_finished(self, summary: str):
        self.progress_label.setText(summary)
        try:
            _app().common_widget_actions.create_and_show_toast_message(self.mw, "Job Manager", summary)
        except Exception:
            pass

    # ---- unattended start -------------------------------------------------
    def _maybe_autostart(self):
        wanted = (
            "--run-jobs" in sys.argv
            or os.environ.get("VISOMASTER_RUN_JOBS", "") == "1"
            or bool(self.store.settings.get("auto_start_on_launch"))
        )
        if not wanted:
            return
        self._autostart_t0 = time.time()
        self.runner.say("Auto-start requested, waiting for the app to finish loading...")
        self._autostart_timer = QtCore.QTimer(self.runner)
        self._autostart_timer.setInterval(1000)
        self._autostart_timer.timeout.connect(self._autostart_tick)
        self._autostart_timer.start()

    def _autostart_tick(self):
        waited = time.time() - self._autostart_t0
        mw = self.mw
        loading = getattr(mw, "_workspace_load_in_progress", False) or not getattr(mw, "_workspace_load_done", True)
        if waited < 5 or (loading and waited < 180):
            return
        self._autostart_timer.stop()
        if loading:
            self.runner.say("The workspace is still loading after 3 minutes, starting anyway.")
        ids = [j["id"] for j in self.store.jobs if j.get("enabled", True) and j["status"] == ST_PENDING]
        if not ids:
            self.runner.say("Auto-start: there are no pending jobs.")
            return
        self.runner.start(ids)


def setup(main_window):
    try:
        return JobManagerExtension(main_window)
    except Exception:
        print("[JOB MANAGER] Could not start:")
        traceback.print_exc()
        return None
