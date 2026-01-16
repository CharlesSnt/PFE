import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mat_to_excel(mat_file_path, output_file_path):
    """Export all non-metadata variables from a .mat file.

    If `output_file_path` ends with `.xlsx` the variables are written as separate
    sheets in a single Excel workbook. Otherwise `output_file_path` is treated
    as a directory (created if necessary) and each variable is written to its own
    CSV file named `<variable>.csv`.
    """
    try:
        mat_data = scipy.io.loadmat(
            mat_file_path, squeeze_me=True, struct_as_record=False
        )
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return
    vars_dict = {k: v for k, v in mat_data.items() if not k.startswith("__")}
    if not vars_dict:
        print("No variables found in the MAT file.")
        return

    def _to_dataframe(name, val):
        arr = np.array(val)
        # Flatten >2D into 2D (rows, flattened columns)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim == 0:
            return pd.DataFrame({name: [arr.item()]})
        elif arr.ndim == 1:
            return pd.DataFrame({name: arr})
        else:
            df = pd.DataFrame(arr)
            # Give deterministic column names
            df.columns = [f"col{c}" for c in range(df.shape[1])]
            return df

    # Write to Excel workbook if requested
    if str(output_file_path).lower().endswith(".xlsx"):
        try:
            with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
                used_sheets = set()
                for name, val in vars_dict.items():
                    df = _to_dataframe(name, val)
                    # sanitize and truncate sheet name to 31 chars
                    sheet = "".join(
                        c for c in name if c.isalnum() or c in (" ", "_", "-")
                    )[:31]
                    if not sheet:
                        sheet = "var"
                    orig = sheet
                    idx = 1
                    while sheet in used_sheets:
                        suffix = f"_{idx}"
                        sheet = (
                            (orig[: 31 - len(suffix)] + suffix)
                            if len(orig) > 31 - len(suffix)
                            else orig + suffix
                        )
                        idx += 1
                    used_sheets.add(sheet)
                    df.to_excel(writer, sheet_name=sheet, index=False)
                    print(f"Wrote variable '{name}' to sheet '{sheet}'")
        except Exception as e:
            print(f"Error writing Excel file: {e}")
            return
    else:
        # Treat output_file_path as directory (or derive one from a .csv base)
        outdir = output_file_path
        if str(output_file_path).lower().endswith(".csv"):
            outdir = os.path.splitext(output_file_path)[0] + "_csv"
        os.makedirs(outdir, exist_ok=True)
        for name, val in vars_dict.items():
            df = _to_dataframe(name, val)
            safe_name = (
                "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name)
                .strip()
                .replace(" ", "_")
            )
            fname = os.path.join(outdir, f"{safe_name}.csv")
            df.to_csv(fname, index=False)
            print(f"Wrote variable '{name}' to '{fname}'")

    # Print a compact preview for each variable
    for name, val in vars_dict.items():
        try:
            _pretty_print_preview(name, np.array(val))
        except Exception:
            pass


def _pretty_print_preview(name, arr, max_rows=10, max_cols=6):
    """Print a compact, nice terminal preview and basic numeric summary for arr."""
    print("=" * 72)
    print(f"Variable: {name}")
    print(
        f"Type: {type(arr)}, dtype: {getattr(arr, 'dtype', None)}, shape: {getattr(arr, 'shape', None)}"
    )
    print("-" * 72)
    try:
        # Ensure numpy array
        arr = np.array(arr)

        # Flatten >2D to 2D for preview
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
            print(f"(Flattened to 2D for preview: {arr.shape})")

        # Make DataFrame for pretty printing
        if arr.ndim == 0:
            print("Scalar value:", arr.item())
            print("=" * 72)
            return
        elif arr.ndim == 1:
            df = pd.DataFrame({"value": arr})
        else:
            df = pd.DataFrame(arr)

        # Print a head with limited columns/rows using pandas display options
        with pd.option_context(
            "display.max_rows",
            max_rows,
            "display.max_columns",
            max_cols,
            "display.width",
            200,
        ):
            print(df.head(max_rows).to_string())

        # If numeric columns exist, show small summary (min/median/mean/max) for first columns
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            stats = num_df.agg(["min", "median", "mean", "max"]).transpose()
            print("\nSummary (first columns shown):")
            print(stats.head(max_cols).to_string(float_format="{:.6g}".format))
    except Exception as e:
        print(f"Could not create preview: {e}")
    print("=" * 72)


def visualize_first_variable(mat_file_path, max_rows=10):
    """Load the MAT file, pick the first non-metadata variable, print a small preview and plot it."""
    try:
        mat_data = scipy.io.loadmat(
            mat_file_path, squeeze_me=True, struct_as_record=False
        )
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    # find first variable that's not metadata
    first_key = None
    first_val = None
    for k, v in mat_data.items():
        if not k.startswith("__"):
            first_key, first_val = k, v
            break

    if first_key is None:
        print("No variables found in the MAT file.")
        return

    arr = np.array(first_val)

    # NEW: pretty terminal preview
    _pretty_print_preview(first_key, arr, max_rows=max_rows, max_cols=6)

    print("Now opening a quick plot window (if plotting makes sense for this data)...")

    try:
        if arr.ndim == 0:
            print("Scalar value:", arr.item())
        elif arr.ndim == 1:
            x = np.arange(arr.shape[0])
            plt.figure(figsize=(8, 4))
            plt.plot(x, arr, marker="o")
            plt.title(f"{first_key} (1D)")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            arr_plot = arr
            if arr_plot.ndim > 2:
                arr_plot = arr_plot.reshape(arr_plot.shape[0], -1)

            rows, cols = arr_plot.shape
            plt.figure(figsize=(8, 4))
            if cols <= 6:
                for c in range(cols):
                    plt.plot(np.arange(rows), arr_plot[:, c], label=f"col{c}")
                plt.legend()
                plt.title(f"{first_key} (2D lines)")
                plt.xlabel("Row")
                plt.ylabel("Value")
            else:
                plt.imshow(arr_plot, aspect="auto", interpolation="nearest")
                plt.colorbar(label="Value")
                plt.title(f"{first_key} (heatmap)")
                plt.xlabel("Column")
                plt.ylabel("Row")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Could not plot variable '{first_key}': {e}")


def visualize_all_variables(mat_file_path, max_rows=10, max_plots=6):
    """Preview and (optionally) plot every variable in the MAT file.

    To avoid opening too many plot windows, `max_plots` limits how many variables
    will be plotted; others will only have a terminal preview printed.
    """
    try:
        mat_data = scipy.io.loadmat(
            mat_file_path, squeeze_me=True, struct_as_record=False
        )
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    vars_found = [(k, v) for k, v in mat_data.items() if not k.startswith("__")]
    if not vars_found:
        print("No variables found in the MAT file.")
        return

    plotted = 0
    for name, val in vars_found:
        arr = np.array(val)
        _pretty_print_preview(name, arr, max_rows=max_rows, max_cols=6)

        if plotted >= max_plots:
            print(f"Skipping plot for '{name}' (plot limit {max_plots} reached)")
            continue

        print(f"Plotting variable '{name}'...")
        try:
            if arr.ndim == 0:
                print("Scalar value:", arr.item())
            elif arr.ndim == 1:
                x = np.arange(arr.shape[0])
                plt.figure(figsize=(8, 4))
                plt.plot(x, arr, marker="o")
                plt.title(f"{name} (1D)")
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                arr_plot = arr
                if arr_plot.ndim > 2:
                    arr_plot = arr_plot.reshape(arr_plot.shape[0], -1)

                rows, cols = arr_plot.shape
                plt.figure(figsize=(8, 4))
                if cols <= 6:
                    for c in range(cols):
                        plt.plot(np.arange(rows), arr_plot[:, c], label=f"col{c}")
                    plt.legend()
                    plt.title(f"{name} (2D lines)")
                    plt.xlabel("Row")
                    plt.ylabel("Value")
                else:
                    plt.imshow(arr_plot, aspect="auto", interpolation="nearest")
                    plt.colorbar(label="Value")
                    plt.title(f"{name} (heatmap)")
                    plt.xlabel("Column")
                    plt.ylabel("Row")
                plt.tight_layout()
                plt.show()
            plotted += 1
        except Exception as e:
            print(f"Could not plot variable '{name}': {e}")


# Usage examples
# Write all variables into an Excel workbook (each variable gets its own sheet)
mat_to_excel("Data//cylinder_nektar_wake.mat", "cylinder_data.xlsx")

# Or write CSV files into a directory:
# mat_to_excel("Data//cylinder_nektar_wake.mat", "output_csv_dir")

# Preview and plot up to N variables (set max_plots to control number of windows)
visualize_all_variables("Data//cylinder_nektar_wake.mat", max_rows=8, max_plots=6)
# If you only want a quick standalone preview of the first variable:
# visualize_first_variable("Data//cylinder_nektar_wake.mat")
