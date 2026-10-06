"""Interactive particle detection in Napari.

Open a 2-D image or a monochrome uint8/uint16 stack with Napari's normal
``File > Open...`` command. Select the image layer in the dock widget and
press ``Detect particles``.

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
    QDoubleSpinBox,
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
    min_area_px: int
    max_area_px: int | None
    opening_kernel_px: int
    opening_iterations: int
    box_size_px: int
    gain: float
    background: int


@dataclass(frozen=True)
class StackLayout:
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


def _stack_layout(data: Any) -> StackLayout:
    """Validate a monochrome uint8/uint16 image or stack."""

    if isinstance(data, (list, tuple)):
        raise ValueError(
            "Multiscale image layers are not supported. "
            "Load a single-resolution image or TIFF stack."
        )

    ndim = int(data.ndim)
    if ndim < 2:
        raise ValueError("The selected layer must contain at least two dimensions.")

    dtype = np.dtype(data.dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise ValueError("Only monochrome uint8 and uint16 images are supported.")

    leading_shape = tuple(int(size) for size in data.shape[: ndim - 2])
    return StackLayout(ndim=ndim, leading_shape=leading_shape)


def _prepare_processing_frame(
    frame: Any,
    settings: DetectionSettings,
) -> np.ndarray:
    """Apply background subtraction and gain without changing the source image."""

    image = np.asarray(frame)

    if image.ndim != 2:
        raise ValueError("Every stack plane must be a 2-D grayscale image.")

    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Only monochrome uint8 and uint16 images are supported.")

    if settings.background == 0 and settings.gain == 1.0:
        return np.ascontiguousarray(image)

    maximum = np.iinfo(image.dtype).max
    processed = (image.astype(np.float32) - settings.background) * settings.gain

    return np.ascontiguousarray(
        np.clip(processed, 0, maximum).astype(image.dtype)
    )


def _rectangles_from_stats(
    stats: np.ndarray,
    centroids: np.ndarray,
    settings: DetectionSettings,
    leading_index: tuple[int, ...],
    ndim: int,
) -> np.ndarray:
    """Create fixed-size squares centred on each accepted particle."""

    component_stats = stats[1:]
    component_centroids = centroids[1:]

    if component_stats.size == 0:
        return np.empty((0, 4, ndim), dtype=np.float32)

    areas = component_stats[:, cv2.CC_STAT_AREA]

    keep = areas >= settings.min_area_px
    if settings.max_area_px is not None:
        keep &= areas <= settings.max_area_px

    component_centroids = component_centroids[keep]
    n_particles = len(component_centroids)

    if n_particles == 0:
        return np.empty((0, 4, ndim), dtype=np.float32)

    half_box = settings.box_size_px / 2.0

    center_x = component_centroids[:, 0]
    center_y = component_centroids[:, 1]

    left = center_x - half_box
    right = center_x + half_box
    top = center_y - half_box
    bottom = center_y + half_box

    rectangles = np.empty((n_particles, 4, ndim), dtype=np.float32)

    if leading_index:
        rectangles[:, :, : len(leading_index)] = np.asarray(
            leading_index,
            dtype=np.float32,
        )

    # Napari coordinates are (..., row, column).
    rectangles[:, 0, -2:] = np.column_stack((top, left))
    rectangles[:, 1, -2:] = np.column_stack((bottom, left))
    rectangles[:, 2, -2:] = np.column_stack((bottom, right))
    rectangles[:, 3, -2:] = np.column_stack((top, right))

    return rectangles


def detect_particles_in_plane(
    frame: Any,
    settings: DetectionSettings,
    leading_index: tuple[int, ...],
    ndim: int,
    ) -> np.ndarray:
    """Detect bright particles in one 2-D image plane."""

    grayscale = _prepare_processing_frame(frame, settings)

    _threshold, mask = cv2.threshold(
    grayscale,0,1,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # connectedComponentsWithStats requires an 8-bit binary mask.
    mask = mask.astype(np.uint8)

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

    _n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
        ltype=cv2.CV_32S,
    )

    return _rectangles_from_stats(
        stats,
        centroids,
        settings,
        leading_index,
        ndim,
    )


@thread_worker(ignore_errors=True)
def detect_stack(
    data: Any,
    settings: DetectionSettings,
):
    """Run detection in a background worker thread."""

    layout = _stack_layout(data)
    total_planes = layout.n_planes
    progress_step = max(1, total_planes // 100)

    rectangles_per_plane: list[np.ndarray] = []
    particles_found = 0
    start = perf_counter()

    for plane_number, leading_index in enumerate(
        np.ndindex(layout.leading_shape),
        start=1,
    ):
        frame = data[leading_index] if leading_index else data

        rectangles = detect_particles_in_plane(
            frame,
            settings=settings,
            leading_index=leading_index,
            ndim=layout.ndim,
        )

        if len(rectangles):
            rectangles_per_plane.append(rectangles)
            particles_found += len(rectangles)

        if plane_number % progress_step == 0 or plane_number == total_planes:
            yield DetectionProgress(
                plane_number,
                total_planes,
                particles_found,
            )

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
    """Particle-detection dock widget for a standalone Napari script."""

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
            "Open an image using <b>File → Open…</b>, select the image layer, "
            "then press <b>Detect particles</b>. "
            "Each bright particle is shown as a fixed-size square."
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

        self.box_size = QSpinBox()
        self.box_size.setRange(1, 10_000)
        self.box_size.setValue(100)
        self.box_size.setSuffix(" px")
        self.box_size.setToolTip(
            "Side length of every displayed particle square."
        )
        parameters.addRow("Square side", self.box_size)

        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.0, 100.0)
        self.gain.setDecimals(3)
        self.gain.setSingleStep(0.1)
        self.gain.setValue(5.0)
        self.gain.setToolTip(
            "Multiplier applied after background subtraction. "
            "The original image is not changed."
        )
        parameters.addRow("Gain", self.gain)

        self.background = QSpinBox()
        self.background.setRange(0, 65_535)
        self.background.setValue(10)
        self.background.setToolTip(
            "Constant subtracted before gain. The original image is not changed."
        )
        parameters.addRow("Background", self.background)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 2_000_000_000)
        self.min_area.setValue(36)
        self.min_area.setSuffix(" px²")
        self.min_area.setToolTip("Components smaller than this are discarded.")
        parameters.addRow("Minimum area", self.min_area)

        self.max_area = QSpinBox()
        self.max_area.setRange(0, 2_000_000_000)
        self.max_area.setValue(0)
        self.max_area.setSpecialValueText("No limit")
        self.max_area.setSuffix(" px²")
        self.max_area.setToolTip(
            "Components larger than this are discarded. "
            "Set to 0 for no upper limit."
        )
        parameters.addRow("Maximum area", self.max_area)

        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(1, 31)
        self.kernel_size.setSingleStep(2)
        self.kernel_size.setValue(1)
        self.kernel_size.setToolTip(
            "Diameter of the opening kernel. "
            "A value of 1 effectively disables morphological opening."
        )
        parameters.addRow("Opening kernel", self.kernel_size)

        self.opening_iterations = QSpinBox()
        self.opening_iterations.setRange(0, 10)
        self.opening_iterations.setValue(1)
        self.opening_iterations.setToolTip(
            "Set to 0 to disable morphological opening."
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
        """Synchronise the layer menu with images opened in Napari."""

        previous_layer = self.image_layer_combo.currentData()
        active_layer = self.viewer.layers.selection.active

        self.image_layer_combo.blockSignals(True)
        self.image_layer_combo.clear()

        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]

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

        # Opening kernels should be odd.
        if kernel % 2 == 0:
            kernel += 1

        max_area = self.max_area.value()

        return DetectionSettings(
            min_area_px=self.min_area.value(),
            max_area_px=max_area if max_area > 0 else None,
            opening_kernel_px=kernel,
            opening_iterations=self.opening_iterations.value(),
            box_size_px=self.box_size.value(),
            gain=self.gain.value(),
            background=self.background.value(),
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
            if source.rgb:
                raise ValueError(
                    "RGB images are not supported. Select a monochrome layer."
                )

            stack_layout = _stack_layout(source.data)

        except (AttributeError, ValueError) as error:
            QMessageBox.warning(self, "Unsupported image", str(error))
            return

        settings = self._settings()

        if (
            settings.max_area_px is not None
            and settings.max_area_px < settings.min_area_px
        ):
            QMessageBox.warning(
                self,
                "Invalid area range",
                "Maximum area must be at least the minimum area, "
                "or set to No limit.",
            )
            return

        maximum = np.iinfo(source.data.dtype).max

        if settings.background > maximum:
            QMessageBox.warning(
                self,
                "Invalid background",
                f"The selected image is {source.data.dtype}; "
                f"background must be between 0 and {maximum}.",
            )
            return

        self._source_context = {
            "id": id(source),
            "name": source.name,
            "scale": tuple(source.scale),
            "translate": tuple(source.translate),
            "ndim": stack_layout.ndim,
        }

        self._set_busy(True)
        self.status.setText(f"Analysing {stack_layout.n_planes} plane(s)…")

        self._worker = detect_stack(source.data, settings)

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
            "edge_width": 2,
            "face_color": np.array(
                [0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
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
            self.viewer.add_shapes(
                ndim=result.ndim,
                **common_arguments,
            )

        rate = (
            result.n_planes / result.elapsed_s
            if result.elapsed_s
            else float("inf")
        )

        self.status.setText(
            f"Detected {result.n_particles} particle(s) in "
            f"{result.n_planes} plane(s) in {result.elapsed_s:.2f} s "
            f"({rate:.1f} plane/s)."
        )

    def _show_error(self, error: BaseException) -> None:
        self.status.setText("Detection failed. See the error message for details.")
        QMessageBox.critical(
            self,
            "Particle detection failed",
            str(error),
        )

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

            if (
                isinstance(layer, Shapes)
                and metadata.get(RESULT_MARKER) is True
            ):
                self.viewer.layers.remove(layer)

        self.status.setText("Particle boxes removed.")


def main() -> None:
    viewer = napari.Viewer(show=True)

    widget = ParticleDetectionWidget(viewer)

    viewer.window.add_dock_widget(
        widget,
        name="Particle detection",
        area="right",
    )

    napari.run()


if __name__ == "__main__":
    main()