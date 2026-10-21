                col3.metric("Sequence length", f"{len(sequence):,} residues")
                st.progress(min(max(probability, 0.0), 1.0))
                st.caption(
                    "The displayed probability is the classifier's sigmoid output; "
                    "it is not necessarily a clinically calibrated probability."
                )
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")


# -------------------- Batch prediction --------------------
with batch_tab:
    st.subheader("Predict multiple sequences")
    st.write(
        "Upload a CSV or Excel file containing a column named `sequence`. "
        "The app adds prediction and score columns."
    )
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            if "sequence" not in batch_df.columns:
                st.error(
                    "Your file must contain a column named `sequence`. "
                    f"Found columns: {', '.join(map(str, batch_df.columns))}"
                )
            else:
                st.write(f"Rows found: {len(batch_df):,}")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if len(batch_df) > 500:
                    st.warning(
                        "Large uploads can take a while. For reliability, process "
                        "500 sequences or fewer per upload."
                    )

                if st.button("Predict uploaded sequences", type="primary"):
                    results = []
                    progress = st.progress(0)
                    status = st.empty()
                    total = len(batch_df)

                    for index, raw_sequence in enumerate(batch_df["sequence"]):
                        sequence = clean_sequence(raw_sequence)
                        problem = validate_sequence(sequence)
                        if problem:
                            results.append(
                                {
                                    "cleaned_sequence": sequence,
                                    "sequence_length": len(sequence),
                                    "allergen_probability": np.nan,
                                    "prediction": "Invalid sequence",
                                    "error": problem,
                                }
                            )
                        else:
                            try:
                                embedding = get_embedding(
                                    sequence, esm_model, batch_converter, device
                                )
                                probability = float(
                                    classifier.predict(embedding, verbose=0)[0][0]
                                )
                                results.append(
                                    {
                                        "cleaned_sequence": sequence,
                                        "sequence_length": len(sequence),
                                        "allergen_probability": probability,
                                        "prediction": (
                                            "Allergenic"
                                            if probability > THRESHOLD
                                            else "Non-Allergenic"
                                        ),
                                        "error": "",
                                    }
                                )
                            except Exception as exc:
                                results.append(
                                    {
                                        "cleaned_sequence": sequence,
                                        "sequence_length": len(sequence),
                                        "allergen_probability": np.nan,
                                        "prediction": "Error",
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                        progress.progress((index + 1) / max(total, 1))
                        status.write(f"Processed {index + 1} of {total}")

                    output_df = pd.concat(
                        [batch_df.reset_index(drop=True), pd.DataFrame(results)],
                        axis=1,
                    )
                    st.success("Batch prediction finished.")
                    st.dataframe(output_df, use_container_width=True)
                    st.download_button(
                        "Download predictions as CSV",
                        data=output_df.to_csv(index=False).encode("utf-8"),
                        file_name="allergen_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not read uploaded file: {type(exc).__name__}: {exc}")


with about_tab:
    st.subheader("How this app works")
    st.markdown(
        """
1. Cleans the input sequence by removing spaces, line breaks, and alignment gaps.
2. Generates a 320-dimensional mean embedding using ESM-2 layer 6.
3. Reshapes the embedding to `(1, 320, 1)`.
4. Runs the saved CNN–BiLSTM classifier.
5. Labels the sequence **Allergenic** when the output is greater than `0.57`;
   otherwise it labels it **Non-Allergenic**.
        """
    )
    st.warning(
        "Predictions depend on the dataset and training procedure used for the "
