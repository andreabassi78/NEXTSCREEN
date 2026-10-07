"""Interactive particle detection in Napari.

Run this file from the VS Code terminal::

    python particle_detection_napari.py

Open a 2-D image or a grayscale/RGB stack with Napari's normal
``File > Open...`` command.  Select the image layer in the dock widget and
press ``Detect particles``.

Dependencies (install in the same Python environment used by VS Code)::

    python -m pip install "napari[all]" opencv-python-headless

The detector is deliberately based on global Otsu thresholding followed by
connected-component labelling.  It is usually faster and lighter than
extracting contours when only particle area and bounding boxes are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import napari
import numpy as np
from napari.layers import Image, Shapes
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


RESULT_MARKER = "particle_detection_napari"


@dataclass(frozen=True)
class DetectionSettings:
    """Parameters used by the threshold-based particle detector."""

    min_area_px: int
    max_area_px: int | None
    opening_kernel_px: int
    opening_iterations: int
    foreground: str


@dataclass(frozen=True)
class StackLayout:
    """The non-colour dimensions of an image stack."""

    ndim: int
    leading_shape: tuple[int, ...]

    @property
    def n_planes(self) -> int:
        return int(np.prod(self.leading_shape, dtype=np.int64)) or 1


@dataclass(frozen=True)
class DetectionProgress:
    plane: int
    total_planes: int
    particles_found: int


@dataclass(frozen=True)
class DetectionResult:
    rectangles: np.ndarray
    ndim: int
    n_planes: int
    n_particles: int
    elapsed_s: float


def _stack_layout(data: Any, is_rgb: bool) -> StackLayout:
    """Validate the array layout and return its stack dimensions.

    Each 2-D plane is analysed independently.  Thus, for a ``(t, y, x)`` or
    ``(z, y, x)`` array, a rectangle is stored at its corresponding ``t`` or
    ``z`` coordinate and appears only on that Napari slice.
    """

    if isinstance(data, (list, tuple)):
        raise ValueError(
            "Multiscale image layers are not supported. Load a single-resolution "
            "image or TIFF stack instead."
        )

    raw_ndim = int(data.ndim)
    logical_ndim = raw_ndim - int(is_rgb)
    if logical_ndim < 2:
        raise ValueError("The selected layer must contain at least two spatial dimensions.")

    if is_rgb and data.shape[-1] not in (3, 4):
        raise ValueError("An RGB image must have 3 or 4 values in its last axis.")

    # In an RGB layer, the trailing RGB(A) axis is not a Napari dimension.
    leading_shape = tuple(int(size) for size in data.shape[: logical_ndim - 2])
    return StackLayout(ndim=logical_ndim, leading_shape=leading_shape)


def _to_grayscale(frame: Any, is_rgb: bool) -> np.ndarray:
    """Return a contiguous 2-D grayscale array suitable for OpenCV."""

    array = np.asarray(frame)
    if is_rgb:
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("Unexpected RGB frame layout.")

        rgb = np.ascontiguousarray(array[..., :3])
        # OpenCV preserves the input depth for these common image types.
        if rgb.dtype in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
            array = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            # This branch is only for uncommon dtypes such as float64 or int32.
            array = (
                0.2126 * rgb[..., 0].astype(np.float32)
                + 0.7152 * rgb[..., 1].astype(np.float32)
                + 0.0722 * rgb[..., 2].astype(np.float32)
            )

    if array.ndim != 2:
        raise ValueError("Every stack plane must be a 2-D grayscale image.")

    if array.dtype == np.uint8 or array.dtype == np.uint16:
        return np.ascontiguousarray(array)

    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8) * 255)

    # cv2.threshold with Otsu accepts uint8 and uint16.  Convert uncommon
    # numerical types once per plane, retaining their full per-plane range.
    finite = np.isfinite(array)
    output = np.zeros(array.shape, dtype=np.uint8)
    if not np.any(finite):
        return output

    valid_values = array[finite]
    low = float(valid_values.min())
    high = float(valid_values.max())
    if high <= low:
        return output

    output[finite] = np.clip(
        (valid_values - low) * (255.0 / (high - low)), 0, 255
    ).astype(np.uint8)
    return np.ascontiguousarray(output)


def _rectangles_from_stats(
    stats: np.ndarray,
    settings: DetectionSettings,
    leading_index: tuple[int, ...],
    ndim: int,
) -> np.ndarray:
    """Convert selected OpenCV component statistics to Napari rectangles."""

    # Row zero is the background component.
    component_stats = stats[1:]
    if component_stats.size == 0:
        return np.empty((0, 4, ndim), dtype=np.float32)

    areas = component_stats[:, cv2.CC_STAT_AREA]
    keep = areas >= settings.min_area_px
    if settings.max_area_px is not None:
        keep &= areas <= settings.max_area_px
    component_stats = component_stats[keep]

    n_particles = len(component_stats)
    if n_particles == 0:
        return np.empty((0, 4, ndim), dtype=np.float32)

    left = component_stats[:, cv2.CC_STAT_LEFT]
    top = component_stats[:, cv2.CC_STAT_TOP]
    right = left + component_stats[:, cv2.CC_STAT_WIDTH]
    bottom = top + component_stats[:, cv2.CC_STAT_HEIGHT]

    rectangles = np.empty((n_particles, 4, ndim), dtype=np.float32)
    if leading_index:
        rectangles[:, :, : len(leading_index)] = np.asarray(
            leading_index, dtype=np.float32
        )

    # Napari coordinate order is (..., row, column), not (x, y).
    rectangles[:, 0, -2:] = np.column_stack((top, left))
    rectangles[:, 1, -2:] = np.column_stack((bottom, left))
    rectangles[:, 2, -2:] = np.column_stack((bottom, right))
    rectangles[:, 3, -2:] = np.column_stack((top, right))
    return rectangles


def detect_particles_in_plane(
    frame: Any,
    is_rgb: bool,
    settings: DetectionSettings,
    leading_index: tuple[int, ...],
    ndim: int,
) -> np.ndarray:
    """Threshold one plane and return one axis-aligned rectangle per particle."""

    grayscale = _to_grayscale(frame, is_rgb)
    threshold_mode = cv2.THRESH_BINARY
    if settings.foreground == "dark":
        threshold_mode = cv2.THRESH_BINARY_INV

    _threshold, mask = cv2.threshold(
        grayscale, 0, 255, threshold_mode | cv2.THRESH_OTSU
    )

    if settings.opening_iterations > 0 and settings.opening_kernel_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (settings.opening_kernel_px, settings.opening_kernel_px),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=settings.opening_iterations,
        )

    # Unlike findContours, this directly provides the area and bounding box.
    _n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    return _rectangles_from_stats(stats, settings, leading_index, ndim)


@thread_worker(ignore_errors=True)
def detect_stack(
    data: Any,
    is_rgb: bool,
    settings: DetectionSettings,
):
    """Run per-plane detection in a worker thread and report bounded progress."""

    layout = _stack_layout(data, is_rgb)
    total_planes = layout.n_planes
    progress_step = max(1, total_planes // 100)
    rectangles_per_plane: list[np.ndarray] = []
    particles_found = 0
    start = perf_counter()

    for plane_number, leading_index in enumerate(np.ndindex(layout.leading_shape), start=1):
        frame = data[leading_index] if leading_index else data
        rectangles = detect_particles_in_plane(
            frame,
            is_rgb=is_rgb,
            settings=settings,
            leading_index=leading_index,
            ndim=layout.ndim,
        )
        if len(rectangles):
            rectangles_per_plane.append(rectangles)
            particles_found += len(rectangles)

        # Yielding periodically keeps the GUI responsive and allows Cancel to
        # stop the generator.  Limiting this to about 100 updates avoids UI
        # overhead on very long stacks.
        if plane_number % progress_step == 0 or plane_number == total_planes:
            yield DetectionProgress(plane_number, total_planes, particles_found)

    if rectangles_per_plane:
        all_rectangles = np.concatenate(rectangles_per_plane, axis=0)
    else:
        all_rectangles = np.empty((0, 4, layout.ndim), dtype=np.float32)

    return DetectionResult(
        rectangles=all_rectangles,
        ndim=layout.ndim,
        n_planes=total_planes,
        n_particles=particles_found,
        elapsed_s=perf_counter() - start,
    )


class ParticleDetectionWidget(QWidget):
    """Standalone dock widget; no Napari plugin packaging is required."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._worker = None
        self._source_context: dict[str, Any] | None = None
        self._build_ui()

        self.viewer.layers.events.inserted.connect(self._refresh_image_layers)
        self.viewer.layers.events.removed.connect(self._refresh_image_layers)
        self._refresh_image_layers()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Open an image with <b>File → Open…</b>, choose its layer, then run "
            "the detector. Each foreground component is shown as a bounding box."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        source_group = QGroupBox("Image source")
        source_form = QFormLayout(source_group)
        self.image_layer_combo = QComboBox()
        source_form.addRow("Image layer", self.image_layer_combo)
        layout.addWidget(source_group)

        parameter_group = QGroupBox("Detection parameters")
        parameters = QFormLayout(parameter_group)

        self.foreground_combo = QComboBox()
        self.foreground_combo.addItem("Bright particles", "bright")
        self.foreground_combo.addItem("Dark particles", "dark")
        parameters.addRow("Foreground", self.foreground_combo)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 2_000_000_000)
        self.min_area.setValue(10)
        self.min_area.setSuffix(" px²")
        self.min_area.setToolTip("Components smaller than this are discarded.")
        parameters.addRow("Minimum area", self.min_area)

        self.max_area = QSpinBox()
        self.max_area.setRange(0, 2_000_000_000)
        self.max_area.setValue(0)
        self.max_area.setSpecialValueText("No limit")
        self.max_area.setSuffix(" px²")
        self.max_area.setToolTip(
            "Components larger than this are discarded; 0 means no upper limit."
        )
        parameters.addRow("Maximum area", self.max_area)

        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(1, 31)
        self.kernel_size.setSingleStep(2)
        self.kernel_size.setValue(3)
        self.kernel_size.setToolTip(
            "Diameter of the elliptical kernel used for morphological opening."
        )
        parameters.addRow("Opening kernel", self.kernel_size)

        self.opening_iterations = QSpinBox()
        self.opening_iterations.setRange(0, 10)
        self.opening_iterations.setValue(1)
        self.opening_iterations.setToolTip(
            "Use 0 to disable morphological opening."
        )
        parameters.addRow("Opening iterations", self.opening_iterations)
        layout.addWidget(parameter_group)

        button_row = QHBoxLayout()
        self.detect_button = QPushButton("Detect particles")
        self.detect_button.clicked.connect(self._start_detection)
        button_row.addWidget(self.detect_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_detection)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.clear_button = QPushButton("Remove particle boxes")
        self.clear_button.clicked.connect(self._clear_results)
        layout.addWidget(self.clear_button)

        self.status = QLabel("Open an image stack to begin.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _refresh_image_layers(self, event: Any = None) -> None:
        """Keep the image-layer selector in sync with File > Open."""

        previous_layer = self.image_layer_combo.currentData()
        active_layer = self.viewer.layers.selection.active

        self.image_layer_combo.blockSignals(True)
        self.image_layer_combo.clear()
        image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        for layer in image_layers:
            self.image_layer_combo.addItem(layer.name, layer)

        preferred_layer = (
            active_layer
            if isinstance(active_layer, Image) and active_layer in image_layers
            else previous_layer
        )
        for index, layer in enumerate(image_layers):
            if layer is preferred_layer:
                self.image_layer_combo.setCurrentIndex(index)
                break
        self.image_layer_combo.blockSignals(False)

    def _selected_image_layer(self) -> Image | None:
        layer = self.image_layer_combo.currentData()
        return layer if isinstance(layer, Image) else None

    def _settings(self) -> DetectionSettings:
        kernel = self.kernel_size.value()
        # An odd kernel is symmetric around its central pixel.
        if kernel % 2 == 0:
            kernel += 1

        max_area = self.max_area.value()
        return DetectionSettings(
            min_area_px=self.min_area.value(),
            max_area_px=max_area if max_area > 0 else None,
            opening_kernel_px=kernel,
            opening_iterations=self.opening_iterations.value(),
            foreground=str(self.foreground_combo.currentData()),
        )

    def _start_detection(self) -> None:
        source = self._selected_image_layer()
        if source is None:
            QMessageBox.information(
                self,
                "No image layer",
                "Open an image with File → Open… and select its image layer.",
            )
            return

        try:
            layout = _stack_layout(source.data, bool(source.rgb))
        except (AttributeError, ValueError) as error:
            QMessageBox.warning(self, "Unsupported image", str(error))
            return

        settings = self._settings()
        if settings.max_area_px is not None and settings.max_area_px < settings.min_area_px:
            QMessageBox.warning(
                self,
                "Invalid area range",
                "Maximum area must be at least the minimum area, or set to No limit.",
            )
            return

        # Store transforms now so that result boxes remain aligned even if the
        # user selects a different layer before processing finishes.
        self._source_context = {
            "id": id(source),
            "name": source.name,
            "scale": tuple(source.scale),
            "translate": tuple(source.translate),
            "ndim": layout.ndim,
        }
        self._set_busy(True)
        self.status.setText(f"Analysing {layout.n_planes} plane(s)…")

        self._worker = detect_stack(source.data, bool(source.rgb), settings)
        self._worker.yielded.connect(self._show_progress)
        self._worker.returned.connect(self._show_result)
        self._worker.errored.connect(self._show_error)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _cancel_detection(self) -> None:
        if self._worker is not None:
            self._worker.quit()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current processing block…")

    def _show_progress(self, progress: DetectionProgress) -> None:
        self.status.setText(
            f"Analysed {progress.plane}/{progress.total_planes} plane(s) — "
            f"{progress.particles_found} particles so far."
        )

    def _remove_results_for_source(self, source_id: int) -> None:
        for layer in list(self.viewer.layers):
            metadata = getattr(layer, "metadata", {}) or {}
            if (
                isinstance(layer, Shapes)
                and metadata.get(RESULT_MARKER) is True
                and metadata.get("source_id") == source_id
            ):
                self.viewer.layers.remove(layer)

    def _show_result(self, result: DetectionResult) -> None:
        if self._source_context is None:
            return

        source = self._source_context
        self._remove_results_for_source(source["id"])
        result_name = f"Particles: {source['name']}"
        common_arguments = {
            "name": result_name,
            "edge_color": "lime",
            "edge_width": 1.5,
            "face_color": (0, 0, 0, 0),
            "scale": source["scale"],
            "translate": source["translate"],
            "metadata": {
                RESULT_MARKER: True,
                "source_id": source["id"],
                "source_name": source["name"],
                "n_particles": result.n_particles,
            },
        }

        if result.n_particles:
            self.viewer.add_shapes(
                result.rectangles,
                shape_type="rectangle",
                **common_arguments,
            )
        else:
            self.viewer.add_shapes(ndim=result.ndim, **common_arguments)

        rate = result.n_planes / result.elapsed_s if result.elapsed_s else float("inf")
        self.status.setText(
            f"Detected {result.n_particles} particle(s) in {result.n_planes} "
            f"plane(s) in {result.elapsed_s:.2f} s ({rate:.1f} plane/s)."
        )

    def _show_error(self, error: BaseException) -> None:
        self.status.setText("Detection failed. See the error message for details.")
        QMessageBox.critical(self, "Particle detection failed", str(error))

    def _worker_finished(self) -> None:
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.detect_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.clear_button.setEnabled(not busy)

    def _clear_results(self) -> None:
        for layer in list(self.viewer.layers):
            metadata = getattr(layer, "metadata", {}) or {}
            if isinstance(layer, Shapes) and metadata.get(RESULT_MARKER) is True:
                self.viewer.layers.remove(layer)
        self.status.setText("Particle boxes removed.")


def main() -> None:
    viewer = napari.Viewer()
    widget = ParticleDetectionWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Particle detection",
        area="right",
    )
    napari.run()


if __name__ == "__main__":
    main()
