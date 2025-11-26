import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import h5py
    import numpy as np
    from bokeh.plotting import figure
    from bokeh.layouts import row
    from bokeh.models import LinearColorMapper
    from bokeh.palettes import Viridis256

    def visualize_cxi_event_bokeh(cxi_path, idx, show_peaks=True, box_size=20):
        """
        Visualize a single event from CXI file using Bokeh.
        """
        print(f"\n=== Visualizing CXI Event {idx} ===")
        print(f"File: {cxi_path}")

        with h5py.File(cxi_path, 'r') as f:
            total_events = len(f['entry_1/data_1/data'])
            print(f"Total events in file: {total_events}")

            if idx >= total_events:
                raise ValueError(f"Index {idx} out of range (0-{total_events-1})")

            detector_image = f['entry_1/data_1/data'][idx]

            has_segmap = '/entry_1/result_1/segmentation_map' in f
            has_logit0 = '/entry_1/result_1/logit_map_class0' in f
            has_logit1 = '/entry_1/result_1/logit_map_class1' in f

            segmentation_map = f['entry_1/result_1/segmentation_map'][idx] if has_segmap else None
            logit_class0 = f['entry_1/result_1/logit_map_class0'][idx] if has_logit0 else None
            logit_class1 = f['entry_1/result_1/logit_map_class1'][idx] if has_logit1 else None

            peak_x = f['entry_1/result_1/peakXPosRaw'][idx]
            peak_y = f['entry_1/result_1/peakYPosRaw'][idx]
            n_peaks_raw = f['entry_1/result_1/nPeaks'][idx]

            valid = (peak_x >= 0) & (peak_y >= 0)
            peak_x = peak_x[valid]
            peak_y = peak_y[valid]

            print(f"\nPeak statistics:")
            print(f"  Number of peaks (from nPeaks): {n_peaks_raw}")
            print(f"  Number of valid peaks: {len(peak_x)}")

        def create_bokeh_panel(image_data, title, vmin, vmax, peak_x, peak_y, show_peaks, box_size):
            h, w = image_data.shape
            mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)

            p = figure(
                title=title,
                x_axis_label="X (pixels)",
                y_axis_label="Y (pixels)",
                tools="pan,box_zoom,wheel_zoom,reset,save",
                width=800,
                height=800,
                match_aspect=True,
            )

            p.image(image=[image_data], x=0, y=0, dw=w, dh=h, color_mapper=mapper)

            if show_peaks and len(peak_x) > 0:
                p.rect(x=peak_x, y=peak_y, width=box_size, height=box_size,
                       fill_alpha=0, line_color="red", line_width=1.5)

            return p

        figures = []

        # Panel 1: Detector image
        vmin = float(detector_image.mean())
        vmax = float(detector_image.mean() + 4 * detector_image.std())
        p1 = create_bokeh_panel(detector_image, f"Detector ({len(peak_x)} peaks)",
                                vmin, vmax, peak_x, peak_y, show_peaks, box_size)
        figures.append(p1)

        # Panel 2: Segmentation map
        if segmentation_map is not None:
            p2 = create_bokeh_panel(segmentation_map, "Segmentation Map",
                                    0, 0.5, peak_x, peak_y, show_peaks, box_size)
            figures.append(p2)

        # Panel 3: Logit class 0
        if logit_class0 is not None:
            vmin = float(logit_class0.mean())
            vmax = float(logit_class0.mean() + 4 * logit_class0.std())
            p3 = create_bokeh_panel(logit_class0, "Logit Class 0 (Background)",
                                    vmin, vmax, peak_x, peak_y, show_peaks, box_size)
            figures.append(p3)

        # Panel 4: Logit class 1
        if logit_class1 is not None:
            vmin = float(logit_class1.mean())
            vmax = float(logit_class1.mean() + 4 * logit_class1.std())
            p4 = create_bokeh_panel(logit_class1, "Logit Class 1 (Peaks)",
                                    vmin, vmax, peak_x, peak_y, show_peaks, box_size)
            figures.append(p4)

        return row(*figures)
    return (visualize_cxi_event_bokeh,)


@app.cell
def _(mo):
    cxi_path = "peaknet_673m_results/peaknet_cxi_20251030_160700_528651_chunk0004.cxi"

    idx = mo.ui.slider(
        start=0,           # minimum value
        stop=30,          # maximum value
        step=1,            # increment
        value=0,          # default value
        label="idx"
    )
    box_size=20
    show_peaks=True
    output_path=None
    show=True
    idx
    return box_size, cxi_path, idx, show_peaks


@app.cell
def _(box_size, cxi_path, idx, mo, show_peaks, visualize_cxi_event_bokeh):
    layout = visualize_cxi_event_bokeh(cxi_path, idx.value, show_peaks, box_size)
    mo.as_html(layout)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
