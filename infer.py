def main(

    camera_index=0,

    checkpoint_coarse="models/laser_heatmap_coarse.pth",

    checkpoint_fine="models/laser_heatmap_fine.pth"

):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    coarse_model = load_model(
        checkpoint_coarse,
        device
    )

    fine_model = load_model(
        checkpoint_fine,
        device
    )

    temp_filter = TemporalHeatmapFilter(alpha=0.8)

    cap = cv2.VideoCapture(camera_index)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        h0, w0 = frame.shape[:2]

        enh = underwater_enhance(frame)

        coarse_img = cv2.resize(
            enh,
            (256, 256)
        )

        coarse_rgb = cv2.cvtColor(
            coarse_img,
            cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        coarse_tensor = torch.tensor(
            np.transpose(coarse_rgb, (2, 0, 1)),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            coarse_heatmap = coarse_model(
                coarse_tensor
            )[0, 0].cpu().numpy()

        coarse_heatmap = temp_filter.update(
            coarse_heatmap
        )

        cx_c, cy_c, conf_c = peak_to_xy(
            coarse_heatmap
        )

        x_orig = int(cx_c * w0 / 256)
        y_orig = int(cy_c * h0 / 256)

        crop, x1, y1 = crop_around_point(
            enh,
            x_orig,
            y_orig,
            crop_size=96
        )

        ch, cw = crop.shape[:2]

        x_final = x_orig
        y_final = y_orig

        conf_f = conf_c

        if ch >= 8 and cw >= 8:

            crop_resized = cv2.resize(
                crop,
                (256, 256)
            )

            crop_rgb = cv2.cvtColor(
                crop_resized,
                cv2.COLOR_BGR2RGB
            ).astype(np.float32) / 255.0

            crop_tensor = torch.tensor(
                np.transpose(crop_rgb, (2, 0, 1)),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                fine_heatmap = fine_model(
                    crop_tensor
                )[0, 0].cpu().numpy()

            fx, fy, conf_f = peak_to_xy(
                fine_heatmap
            )

            x_final = int(
                x1 + fx * cw / 256
            )

            y_final = int(
                y1 + fy * ch / 256
            )

        conf = (
            0.4 * conf_c
            +
            0.6 * conf_f
        )

        vis = frame.copy()

        cv2.circle(
            vis,
            (x_final, y_final),
            8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Laser Detection",
            vis
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()
    if __name__ == "__main__":
        main()