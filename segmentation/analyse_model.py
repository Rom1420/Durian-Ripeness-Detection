import os
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copyfile

YOLOV8_SEG_FOLDER = "model/yolov8m-seg"


def analyse_yolov8_segmentation(model_run_folder):
    model_name = os.path.basename(os.path.normpath(model_run_folder))
    analysis_folder = f"analysis_{model_name}"
    os.makedirs(analysis_folder, exist_ok=True)

    csv_path = os.path.join(model_run_folder, "results.csv")

    # Copy image files to analysis folder for reuse (if present)
    image_files = [
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
        "BoxR_curve.png",
        "confusion_matrix_normalised.png",
        "confusion_matrix.png",
        "labels_correlogram.jpg",
        "labels.jpg",
        "MaskF1_curve.png",
        "MaskP_curve.png",
        "MaskPR_curve.png",
        "MaskR_curve.png",
        "results.png",
    ]
    for img_file in image_files:
        src = os.path.join(model_run_folder, img_file)
        if os.path.exists(src):
            dst = os.path.join(analysis_folder, img_file)
            copyfile(src, dst)

    if os.path.exists(csv_path):
        print(f"Reading metrics file: {csv_path}")
        metrics_df = pd.read_csv(csv_path)
        print(f"Columns found: {metrics_df.columns.tolist()}")

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Metrics Analysis - {model_name}")

        # Total training loss
        if all(
            col in metrics_df.columns
            for col in [
                "train/box_loss",
                "train/seg_loss",
                "train/cls_loss",
                "train/dfl_loss",
            ]
        ):
            total_train_loss = metrics_df[
                ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"]
            ].sum(axis=1)
            axs[0, 0].plot(
                metrics_df["epoch"], total_train_loss, label="Train Total Loss"
            )
            axs[0, 0].set_title("Train Total Loss")
            axs[0, 0].set_xlabel("Epoch")
            axs[0, 0].set_ylabel("Loss")
            axs[0, 0].legend()

        # mAP@0.5 for bounding boxes
        if "metrics/mAP50(B)" in metrics_df.columns:
            axs[0, 1].plot(
                metrics_df["epoch"],
                metrics_df["metrics/mAP50(B)"],
                label="mAP@0.5 (Bounding Boxes)",
                color="green",
            )
            axs[0, 1].set_title("mAP@0.5 (Bounding Boxes)")
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("mAP")
            axs[0, 1].legend()

        # Precision (Bounding boxes)
        if "metrics/precision(B)" in metrics_df.columns:
            axs[1, 0].plot(
                metrics_df["epoch"],
                metrics_df["metrics/precision(B)"],
                label="Precision (Bounding Boxes)",
                color="orange",
            )
            axs[1, 0].set_title("Precision (Bounding Boxes)")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Precision")
            axs[1, 0].legend()

        # Recall (Bounding boxes)
        if "metrics/recall(B)" in metrics_df.columns:
            axs[1, 1].plot(
                metrics_df["epoch"],
                metrics_df["metrics/recall(B)"],
                label="Recall (Bounding Boxes)",
                color="red",
            )
            axs[1, 1].set_title("Recall (Bounding Boxes)")
            axs[1, 1].set_xlabel("Epoch")
            axs[1, 1].set_ylabel("Recall")
            axs[1, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(analysis_folder, "metrics_curves.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plots saved at: {plot_path}")

        # Extract summary metrics
        best_map = (
            metrics_df["metrics/mAP50(B)"].max()
            if "metrics/mAP50(B)" in metrics_df.columns
            else None
        )
        best_map_epoch = (
            metrics_df.loc[metrics_df["metrics/mAP50(B)"].idxmax(), "epoch"]
            if best_map is not None
            else None
        )

        final_loss = (
            total_train_loss.iloc[-1] if "total_train_loss" in locals() else None
        )
        final_precision = (
            metrics_df["metrics/precision(B)"].iloc[-1]
            if "metrics/precision(B)" in metrics_df.columns
            else None
        )
        final_recall = (
            metrics_df["metrics/recall(B)"].iloc[-1]
            if "metrics/recall(B)" in metrics_df.columns
            else None
        )

        summary = (
            f"""
Summary of metrics for {model_name}:

- Best mAP@0.5 (Bounding Boxes): {best_map:.4f} at epoch {best_map_epoch}
- Final total train loss: {final_loss:.4f}
- Final Precision (Bounding Boxes): {final_precision:.4f}
- Final Recall (Bounding Boxes): {final_recall:.4f}
"""
            if all(
                v is not None
                for v in [
                    best_map,
                    best_map_epoch,
                    final_loss,
                    final_precision,
                    final_recall,
                ]
            )
            else "Incomplete data for summary."
        )

        with open(
            os.path.join(analysis_folder, "summary.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(summary)

        print("Summary saved.")

    else:
        print("No metrics file found.")


if __name__ == "__main__":
    analyse_yolov8_segmentation(YOLOV8_SEG_FOLDER)
