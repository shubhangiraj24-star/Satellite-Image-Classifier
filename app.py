# =========================
# GRAD-CAM
# =========================
st.markdown("## Grad-CAM Visualization")

try:

    base_model = model.layers[0]

    last_conv_layer = base_model.get_layer("top_conv")

    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:

        conv_outputs = feature_model(img_array)

        tape.watch(conv_outputs)

        x = base_model.get_layer("top_bn")(conv_outputs)

        x = base_model.get_layer("top_activation")(x)

        for layer in model.layers[1:]:

            x = layer(x)

        preds = x

        class_channel = preds[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # SAFETY CHECK
    if grads is None:

        st.error("Gradients could not be computed.")

    else:

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)

        # AVOID DIVISION BY ZERO
        max_val = np.max(heatmap)

        if max_val != 0:

            heatmap /= max_val

        else:

            heatmap = np.zeros_like(heatmap)

        # =========================
        # RESIZE HEATMAP
        # =========================
        heatmap = cv2.resize(
            heatmap.numpy(),
            (
                original_img.shape[1],
                original_img.shape[0]
            )
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # =========================
        # OVERLAY
        # =========================
        superimposed_img = cv2.addWeighted(
            original_img,
            0.6,
            heatmap,
            0.4,
            0
        )

        # =========================
        # DISPLAY IMAGES
        # =========================
        col1, col2 = st.columns(2)

        with col1:

            st.image(
                heatmap,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with col2:

            st.image(
                superimposed_img,
                caption="AI Attention Map",
                use_container_width=True
            )

except Exception as e:

    st.error(f"Grad-CAM Error: {e}")