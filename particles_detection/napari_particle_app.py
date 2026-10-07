"""Run the standalone HDF5 loader and particle-detection Napari widgets.

From the VS Code terminal, run::

    python napari_particle_app.py
"""

import napari

from h5_stack_opener_napari import H5StackOpenerWidget
from particle_detection_napari import ParticleDetectionWidget


def main() -> None:
    viewer = napari.Viewer(show=True)

    h5_widget = H5StackOpenerWidget(viewer)
    viewer.window.add_dock_widget(
        h5_widget,
        name="Open H5 stacks",
        area="right",
    )

    particle_widget = ParticleDetectionWidget(viewer)
    viewer.window.add_dock_widget(
        particle_widget,
        name="Particle detection",
        area="right",
    )

    napari.run()


if __name__ == "__main__":
    main()
