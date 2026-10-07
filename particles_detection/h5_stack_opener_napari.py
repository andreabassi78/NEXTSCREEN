"""HDF5 stack loader used by the standalone Napari particle-detection app.

This widget is designed for HDF5 files written by the ScopeFoundry
``save_stack`` function.  In that format, each channel is stored as a dataset
such as ``.../t0/c0/stack`` or ``.../t0/c1/stack`` with shape ``(z, y, x)``.

It also supports a single image dataset with a channel axis.  If the HDF5
metadata identifies that axis (for example ``axes='CZYX'``), it is split
automatically.  Otherwise choose the channel axis explicitly in the widget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


STACK_DATASET_NAME = "stack"
MEMORY_WARNING_BYTES = 1_000_000_000


@dataclass(frozen=True)
class H5DatasetInfo:
    """Lightweight description of one image-like HDF5 dataset."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    axes: str | None


@dataclass(frozen=True)
class H5LoadProgress:
    completed_datasets: int
    total_datasets: int
    loaded_layers: int


@dataclass(frozen=True)
class H5ImageLayer:
    """Data and Napari properties for one image layer."""

    data: np.ndarray
    name: str
    scale: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class H5LoadResult:
    layers: tuple[H5ImageLayer, ...]


def _as_text(value: Any) -> str | None:
    """Convert a scalar HDF5 text attribute to a normal Python string."""

    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _as_text(value.item())
    return None


def _axes_from_dataset(dataset: h5py.Dataset) -> str | None:
    """Read common HDF5 axis annotations, if present."""

    axes = _as_text(dataset.attrs.get("axes"))
    if axes and len(axes) == dataset.ndim:
        return axes

    labels: list[str] = []
    for dimension in dataset.dims:
        label = getattr(dimension, "label", "")
        if not label:
            return None
        labels.append(str(label))

    if len(labels) == dataset.ndim:
        return ",".join(labels)
    return None


def _find_image_datasets(file_path: str) -> list[H5DatasetInfo]:
    """List all numeric datasets with at least two dimensions, without loading them."""

    datasets: list[H5DatasetInfo] = []

    with h5py.File(file_path, "r") as h5_file:

        def visit(name: str, item: Any) -> None:
            if not isinstance(item, h5py.Dataset):
                return
            if item.ndim < 2 or item.dtype.kind not in "buif":
                return

            datasets.append(
                H5DatasetInfo(
                    path=name,
                    shape=tuple(int(size) for size in item.shape),
                    dtype=str(item.dtype),
                    nbytes=int(item.size * item.dtype.itemsize),
                    axes=_axes_from_dataset(item),
                )
            )

        h5_file.visititems(visit)

    return sorted(datasets, key=lambda dataset: dataset.path)


def _dataset_leaf_name(dataset_path: str) -> str:
    return dataset_path.rsplit("/", maxsplit=1)[-1]


def _format_bytes(nbytes: int) -> str:
    """Return a compact human-readable byte count."""

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(nbytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _channel_axis_from_metadata(dataset: h5py.Dataset) -> int | None:
    """Return a channel axis only when it is explicitly described by metadata."""

    axes = _axes_from_dataset(dataset)
    if axes is None:
        return None

    compact_axes = axes.replace(",", "").replace(" ", "").upper()
    if len(compact_axes) == dataset.ndim and "C" in compact_axes:
        return compact_axes.index("C")

    labels = [part.strip().lower() for part in axes.split(",")]
    for axis, label in enumerate(labels):
        if label in {"c", "channel", "channels"}:
            return axis
    return None


def _resolve_channel_axis(
    dataset: h5py.Dataset,
    channel_axis_setting: str | int,
) -> int | None:
    """Resolve the UI setting for the channel axis of one dataset."""

    if channel_axis_setting == "auto":
        channel_axis = _channel_axis_from_metadata(dataset)
    elif channel_axis_setting == "none":
        channel_axis = None
    else:
        channel_axis = int(channel_axis_setting)

    if channel_axis is None:
        return None

    # The final two dimensions are interpreted as (y, x), so they cannot be a
    # channel axis.  An explicit invalid setting is reported rather than guessed.
    if channel_axis < 0 or channel_axis >= dataset.ndim - 2:
        raise ValueError(
            f"Dataset '{dataset.name}' has shape {dataset.shape}; "
            f"axis {channel_axis} cannot be used as a channel axis."
        )
    return channel_axis


def _scale_from_dataset(dataset: h5py.Dataset, output_ndim: int) -> tuple[float, ...]:
    """Convert ScopeFoundry's ``element_size_um`` attribute to a Napari scale."""

    raw_scale = dataset.attrs.get("element_size_um")
    if raw_scale is None:
        return (1.0,) * output_ndim

    try:
        values = tuple(float(value) for value in np.ravel(raw_scale))
    except (TypeError, ValueError):
        return (1.0,) * output_ndim

    # ScopeFoundry saves (z, y, x).  Any additional leading dimension in the
    # loaded data is a non-spatial index and therefore gets scale 1.
    values = values[-output_ndim:]
    if len(values) < output_ndim:
        values = (1.0,) * (output_ndim - len(values)) + values
    return values


def _layer_name(
    file_path: str,
    dataset_path: str,
    channel_index: int | None = None,
) -> str:
    """Create a readable, unique-enough Napari layer name."""

    ancestors = dataset_path.split("/")[-3:-1]
    location = " / ".join(ancestors) if ancestors else dataset_path
    name = f"{Path(file_path).stem} | {location}"
    if channel_index is not None:
        name += f" | channel {channel_index}"
    return name


def _make_image_layers(
    file_path: str,
    dataset_path: str,
    dataset: h5py.Dataset,
    channel_axis_setting: str | int,
) -> list[H5ImageLayer]:
    """Read one dataset and return one Napari layer per channel, if needed."""

    channel_axis = _resolve_channel_axis(dataset, channel_axis_setting)
    data = np.ascontiguousarray(dataset[...])

    common_metadata = {
        "h5_file": str(Path(file_path).resolve()),
        "h5_dataset": dataset_path,
        "h5_source_shape": tuple(int(size) for size in dataset.shape),
        "h5_axes": _axes_from_dataset(dataset),
    }

    if channel_axis is None:
        return [
            H5ImageLayer(
                data=data,
                name=_layer_name(file_path, dataset_path),
                scale=_scale_from_dataset(dataset, data.ndim),
                metadata=common_metadata,
            )
        ]

    image_layers: list[H5ImageLayer] = []
    for channel_index in range(data.shape[channel_axis]):
        channel_data = np.ascontiguousarray(
            np.take(data, channel_index, axis=channel_axis)
        )
        metadata = {
            **common_metadata,
            "h5_channel_axis": channel_axis,
            "h5_channel_index": channel_index,
        }
        image_layers.append(
            H5ImageLayer(
                data=channel_data,
                name=_layer_name(file_path, dataset_path, channel_index),
                scale=_scale_from_dataset(dataset, channel_data.ndim),
                metadata=metadata,
            )
        )
    return image_layers


@thread_worker(ignore_errors=True)
def load_h5_datasets(
    file_path: str,
    dataset_paths: tuple[str, ...],
    channel_axis_setting: str | int,
):
    """Load HDF5 datasets in a worker thread, keeping the Napari GUI responsive."""

    image_layers: list[H5ImageLayer] = []
    total_datasets = len(dataset_paths)

    with h5py.File(file_path, "r") as h5_file:
        for completed_datasets, dataset_path in enumerate(dataset_paths, start=1):
            dataset = h5_file[dataset_path]
            image_layers.extend(
                _make_image_layers(
                    file_path,
                    dataset_path,
                    dataset,
                    channel_axis_setting,
                )
            )
            yield H5LoadProgress(
                completed_datasets=completed_datasets,
                total_datasets=total_datasets,
                loaded_layers=len(image_layers),
            )

    return H5LoadResult(layers=tuple(image_layers))


class H5StackOpenerWidget(QWidget):
    """Dock widget that opens HDF5 image stacks without requiring a plugin."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._file_path: str | None = None
        self._datasets: dict[str, H5DatasetInfo] = {}
        self._worker = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        file_group = QGroupBox("HDF5 file")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected.")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        self.choose_file_button = QPushButton("Choose H5 file…")
        self.choose_file_button.clicked.connect(self._choose_file)
        file_layout.addWidget(self.choose_file_button)
        layout.addWidget(file_group)

        dataset_group = QGroupBox("Image datasets")
        dataset_layout = QVBoxLayout(dataset_group)
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dataset_list.itemSelectionChanged.connect(self._update_channel_axis_choices)
        dataset_layout.addWidget(self.dataset_list)
        layout.addWidget(dataset_group)

        options_group = QGroupBox("Multichannel data")
        options_form = QFormLayout(options_group)
        self.channel_axis = QComboBox()
        self.channel_axis.setToolTip(
            "Use Automatic when the HDF5 dataset contains an explicit axis "
            "annotation such as CZYX. For an unannotated CZYX dataset, choose Axis 0."
        )
        options_form.addRow("Channel axis", self.channel_axis)
        layout.addWidget(options_group)

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load selected stacks")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self._start_loading)
        button_row.addWidget(self.load_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_loading)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.status = QLabel("Choose an H5 file to inspect its datasets.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._update_channel_axis_choices()

    def _choose_file(self) -> None:
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 image stack",
            "",
            "HDF5 files (*.h5 *.hdf5 *.hdf);;All files (*)",
        )
        if not file_path:
            return

        try:
            datasets = _find_image_datasets(file_path)
        except OSError as error:
            QMessageBox.critical(
                self,
                "Cannot open HDF5 file",
                str(error),
            )
            return

        if not datasets:
            QMessageBox.information(
                self,
                "No image datasets",
                "The file contains no numeric dataset with at least two dimensions.",
            )
            return

        self._file_path = file_path
        self._datasets = {dataset.path: dataset for dataset in datasets}
        self.file_label.setText(file_path)
        self.dataset_list.clear()

        stack_paths = {
            dataset.path
            for dataset in datasets
            if _dataset_leaf_name(dataset.path).lower() == STACK_DATASET_NAME
        }

        for dataset in datasets:
            item = QListWidgetItem(
                f"{dataset.path}    {dataset.shape}    {dataset.dtype}    "
                f"({_format_bytes(dataset.nbytes)})"
            )
            item.setData(Qt.UserRole, dataset.path)
            self.dataset_list.addItem(item)

            if dataset.path in stack_paths:
                item.setSelected(True)

        # Fall back to the first image-like dataset for H5 files that do not
        # use the exact ScopeFoundry dataset name.
        if not stack_paths:
            self.dataset_list.setCurrentRow(0)
            self.dataset_list.item(0).setSelected(True)

        self.load_button.setEnabled(True)
        self._update_channel_axis_choices()

        selected_count = len(self._selected_dataset_paths())
        self.status.setText(
            f"Found {len(datasets)} image dataset(s); "
            f"selected {selected_count} dataset(s)."
        )

    def _selected_dataset_paths(self) -> tuple[str, ...]:
        return tuple(
            item.data(Qt.UserRole)
            for item in self.dataset_list.selectedItems()
        )

    def _update_channel_axis_choices(self) -> None:
        previous_value = self.channel_axis.currentData()
        selected_datasets = [
            self._datasets[path]
            for path in self._selected_dataset_paths()
            if path in self._datasets
        ]

        maximum_ndim = max(
            (len(dataset.shape) for dataset in selected_datasets),
            default=2,
        )
        self.channel_axis.blockSignals(True)
        self.channel_axis.clear()
        self.channel_axis.addItem("Automatic (metadata only)", "auto")
        self.channel_axis.addItem("No channel axis", "none")

        # The last two axes are always interpreted as (y, x).
        for axis in range(maximum_ndim - 2):
            self.channel_axis.addItem(f"Axis {axis}", axis)

        for index in range(self.channel_axis.count()):
            if self.channel_axis.itemData(index) == previous_value:
                self.channel_axis.setCurrentIndex(index)
                break
        self.channel_axis.blockSignals(False)

    def _start_loading(self) -> None:
        if self._file_path is None:
            return

        dataset_paths = self._selected_dataset_paths()
        if not dataset_paths:
            QMessageBox.information(
                self,
                "No dataset selected",
                "Select one or more image datasets before loading.",
            )
            return

        total_bytes = sum(self._datasets[path].nbytes for path in dataset_paths)
        if total_bytes >= MEMORY_WARNING_BYTES:
            answer = QMessageBox.question(
                self,
                "Large HDF5 data",
                f"The selected datasets occupy about {_format_bytes(total_bytes)} "
                "in memory. Loading copies them into RAM. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._set_busy(True)
        self.status.setText(f"Loading {len(dataset_paths)} dataset(s)…")

        self._worker = load_h5_datasets(
            self._file_path,
            dataset_paths,
            self.channel_axis.currentData(),
        )
        self._worker.yielded.connect(self._show_progress)
        self._worker.returned.connect(self._show_result)
        self._worker.errored.connect(self._show_error)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _cancel_loading(self) -> None:
        if self._worker is not None:
            self._worker.quit()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current dataset…")

    def _show_progress(self, progress: H5LoadProgress) -> None:
        self.status.setText(
            f"Loaded {progress.completed_datasets}/{progress.total_datasets} "
            f"dataset(s) as {progress.loaded_layers} image layer(s)."
        )

    def _show_result(self, result: H5LoadResult) -> None:
        for layer in result.layers:
            self.viewer.add_image(
                layer.data,
                name=layer.name,
                scale=layer.scale,
                metadata=layer.metadata,
            )

        self.status.setText(
            f"Loaded {len(result.layers)} image layer(s). "
            "Select one in the Particle detection widget."
        )

    def _show_error(self, error: BaseException) -> None:
        self.status.setText("HDF5 loading failed. See the error message for details.")
        QMessageBox.critical(self, "HDF5 loading failed", str(error))

    def _worker_finished(self) -> None:
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.choose_file_button.setEnabled(not busy)
        self.dataset_list.setEnabled(not busy)
        self.channel_axis.setEnabled(not busy)
        self.load_button.setEnabled(not busy and bool(self._datasets))
        self.cancel_button.setEnabled(busy)

def main() -> None:
    import napari
    viewer = napari.Viewer(show=True)

    h5_widget = H5StackOpenerWidget(viewer)
    viewer.window.add_dock_widget(
        h5_widget,
        name="Open H5 stacks",
        area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()