"""
streamlit_app.py
----------------
Main entry point for the Demand Forecasting SaaS platform built with
Streamlit. Orchestrates the full end-to-end pipeline from CSV upload to
interactive forecast dashboard.
"""

import os
import sys

# Ensure src/ is importable regardless of the working directory
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src import data_validation, preprocessing, feature_engineering, forecasting, storage

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Demand Forecasting Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    """Initialise all Streamlit session state keys on first load."""
    defaults = {
        "df_raw": None,
        "df_mapped": None,
        "df_features": None,
        "forecast_df": None,
        "prophet_metrics": None,
        "xgb_metrics": None,
        "best_model_name": None,
        "uploaded_filename": None,
        "forecast_ready": False,
        "forecast_filepath": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Sidebar — Previous Uploads
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Previous Uploads")
    upload_log = storage.load_upload_log()
    if not upload_log:
        st.info("No previous uploads")
    else:
        recent = upload_log[-5:][::-1]  # Show newest first, max 5
        for entry in recent:
            st.markdown(
                f"**{entry['filename']}**  \n"
                f"🕒 {entry['timestamp']}  \n"
                f"Rows: {entry['row_count']:,}"
            )
            st.divider()

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("Demand Forecasting Platform")
st.caption("Upload your sales data, configure columns, and generate a 42-day forecast.")

# ---------------------------------------------------------------------------
# SECTION 1 — File Upload
# ---------------------------------------------------------------------------
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing your historical sales data",
    type=["csv"],
    help="Use the template in templates/dataset_template.csv as a guide.",
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state["df_raw"] = df_raw
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df_raw.head(), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")

# ---------------------------------------------------------------------------
# SECTION 2 — Column Mapping
# ---------------------------------------------------------------------------
if st.session_state["df_raw"] is not None:
    st.header("2. Column Mapping")
    df_raw = st.session_state["df_raw"]
    all_cols = list(df_raw.columns)

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox(
            "Select Date Column",
            options=all_cols,
            help="Column containing date values (YYYY-MM-DD).",
        )
        sales_col = st.selectbox(
            "Select Sales Quantity Column",
            options=all_cols,
            help="Column containing numeric sales or demand figures.",
        )
        product_col = st.selectbox(
            "Select Product Column (optional)",
            options=["None"] + all_cols,
            help="Column identifying the product. Leave as None for a single series.",
        )
        store_col = st.selectbox(
            "Select Store Column (optional)",
            options=["None"] + all_cols,
            help="Column identifying the store/location.",
        )

    with col2:
        promo_col = st.selectbox(
            "Select Promotion Column (optional)",
            options=["None"] + all_cols,
            help="Binary column (0/1) indicating promotion days.",
        )
        holiday_col = st.selectbox(
            "Select Holiday Column (optional)",
            options=["None"] + all_cols,
            help="Binary column (0/1) indicating public holidays.",
        )
        price_col = st.selectbox(
            "Select Price Column (optional)",
            options=["None"] + all_cols,
            help="Numeric column containing the product price.",
        )

    # Build the column-rename mapping
    rename_map = {date_col: "date", sales_col: "sales_qty"}
    if product_col != "None":
        rename_map[product_col] = "product_id"
    if store_col != "None":
        rename_map[store_col] = "store_id"
    if promo_col != "None":
        rename_map[promo_col] = "is_promotion"
    if holiday_col != "None":
        rename_map[holiday_col] = "holiday_flag"
    if price_col != "None":
        rename_map[price_col] = "price"

    # Apply mapping and keep only standard-name columns
    try:
        df_mapped = df_raw.rename(columns=rename_map)
        standard_cols = ["date", "sales_qty", "product_id", "store_id",
                         "is_promotion", "holiday_flag", "price"]
        keep_cols = [c for c in standard_cols if c in df_mapped.columns]
        df_mapped = df_mapped[keep_cols]
        st.session_state["df_mapped"] = df_mapped
        st.success("Column mapping complete")
    except Exception as exc:
        st.error(f"Column mapping failed: {exc}")

# ---------------------------------------------------------------------------
# SECTION 3 — Dataset Summary
# ---------------------------------------------------------------------------
if st.session_state["df_mapped"] is not None:
    st.header("3. Dataset Summary")
    try:
        summary = data_validation.detect_optional_columns(
            data_validation.validate_dataset(st.session_state["df_mapped"])
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{summary['total_rows']:,}")
        m2.metric("Date Range (days)", f"{summary['date_range_days']:,}")
        m3.metric(
            "Unique Products",
            str(summary["unique_products"]) if summary["unique_products"] is not None else "N/A",
        )
        m4.metric(
            "Unique Stores",
            str(summary["unique_stores"]) if summary["unique_stores"] is not None else "N/A",
        )
    except Exception as exc:
        st.error(f"Could not compute dataset summary: {exc}")

# ---------------------------------------------------------------------------
# SECTION 4 — Run Forecast
# ---------------------------------------------------------------------------
if st.session_state["df_mapped"] is not None:
    st.header("4. Run Forecast")

    if st.button("Run Forecast", type="primary"):
        st.session_state["forecast_ready"] = False
        error_occurred = False

        try:
            # Step 1 — Validate
            with st.spinner("Step 1/7 — Validating dataset..."):
                df_validated = data_validation.validate_dataset(
                    st.session_state["df_mapped"]
                )
        except Exception as exc:
            st.error(f"[Validation] {exc}")
            error_occurred = True

        if not error_occurred:
            try:
                # Step 2 — Clean
                with st.spinner("Step 2/7 — Cleaning data..."):
                    df_clean = preprocessing.clean_dataframe(df_validated)
                    storage.save_processed_data(
                        df_clean, st.session_state["uploaded_filename"]
                    )
            except Exception as exc:
                st.error(f"[Preprocessing] {exc}")
                error_occurred = True

        if not error_occurred:
            try:
                # Step 3 — Features
                with st.spinner("Step 3/7 — Engineering features..."):
                    df_features = feature_engineering.create_all_features(df_clean)
                    st.session_state["df_features"] = df_features
            except Exception as exc:
                st.error(f"[Feature Engineering] {exc}")
                error_occurred = True

        if not error_occurred:
            try:
                # Step 4 — Prophet
                with st.spinner("Step 4/7 — Training Prophet model..."):
                    prophet_model, prophet_metrics = forecasting.train_prophet(df_features)
                    st.session_state["prophet_metrics"] = prophet_metrics
                    storage.save_model(prophet_model, "Prophet")
            except Exception as exc:
                st.error(f"[Prophet Training] {exc}")
                error_occurred = True

        if not error_occurred:
            try:
                # Step 5 — XGBoost
                with st.spinner("Step 5/7 — Training XGBoost model..."):
                    xgb_model, xgb_metrics, feature_cols = forecasting.train_xgboost(df_features)
                    st.session_state["xgb_metrics"] = xgb_metrics
                    storage.save_model(xgb_model, "XGBoost")
            except Exception as exc:
                st.error(f"[XGBoost Training] {exc}")
                error_occurred = True

        if not error_occurred:
            try:
                # Step 6 — Select best
                with st.spinner("Step 6/7 — Selecting best model..."):
                    results = {
                        "Prophet": st.session_state["prophet_metrics"],
                        "XGBoost": xgb_metrics,
                    }
                    best_name = forecasting.select_best_model(results)
                    st.session_state["best_model_name"] = best_name
                    best_model = prophet_model if best_name == "Prophet" else xgb_model
                    best_feature_cols = [] if best_name == "Prophet" else feature_cols
            except Exception as exc:
                st.error(f"[Model Selection] {exc}")
                error_occurred = True

        if not error_occurred:
            try:
                # Step 7 — Forecast
                with st.spinner("Step 7/7 — Generating forecast..."):
                    forecast_df = forecasting.generate_forecast(
                        best_model,
                        df_features,
                        best_name,
                        best_feature_cols,
                        horizon=42,
                    )
                    st.session_state["forecast_df"] = forecast_df

                    filepath = storage.save_forecast_output(
                        forecast_df, st.session_state["uploaded_filename"]
                    )
                    st.session_state["forecast_filepath"] = filepath
                    st.session_state["forecast_ready"] = True
                    st.success(
                        f"Forecast complete! Best model: **{best_name}** "
                        f"(MAPE {results[best_name]['MAPE']:.2f}%)"
                    )
            except Exception as exc:
                st.error(f"[Forecast Generation] {exc}")

# ---------------------------------------------------------------------------
# SECTION 5 — Dashboard
# ---------------------------------------------------------------------------

# Shared Plotly config — enables full toolbar + scroll zoom in every chart
_PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

if st.session_state["forecast_ready"]:
    st.header("5. Dashboard")
    df_features = st.session_state["df_features"]
    forecast_df = st.session_state["forecast_df"]
    prophet_metrics = st.session_state["prophet_metrics"]
    xgb_metrics = st.session_state["xgb_metrics"]
    best_name = st.session_state["best_model_name"]

    tab_trends, tab_forecast, tab_compare = st.tabs(
        ["📈 Sales Trends", "🔮 Forecast", "⚖️ Model Comparison"]
    )

    # ── Tab 1: Sales Trends ──────────────────────────────────────────────
    with tab_trends:
        st.subheader("Historical Sales Trends")

        plot_df = df_features.copy()

        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])

        # Optional product filter
        with filter_col1:
            if "product_id" in plot_df.columns:
                products = sorted(plot_df["product_id"].dropna().unique().tolist())
                selected_product = st.selectbox(
                    "Filter by Product", options=["All"] + [str(p) for p in products]
                )
                if selected_product != "All":
                    plot_df = plot_df[plot_df["product_id"].astype(str) == selected_product]

        # Optional store filter
        with filter_col2:
            if "store_id" in plot_df.columns:
                stores = sorted(plot_df["store_id"].dropna().unique().tolist())
                selected_store = st.selectbox(
                    "Filter by Store", options=["All"] + [str(s) for s in stores]
                )
                if selected_store != "All":
                    plot_df = plot_df[plot_df["store_id"].astype(str) == selected_store]

        with filter_col3:
            show_ma = st.checkbox("Show 7-day moving average", value=True)
            show_promo = st.checkbox(
                "Highlight promotion days",
                value=False,
                disabled="is_promotion" not in plot_df.columns,
            )

        agg_df = plot_df.groupby("date")["sales_qty"].sum().reset_index()
        agg_df = agg_df.sort_values("date")
        agg_df["ma7"] = agg_df["sales_qty"].rolling(7, min_periods=1).mean()
        agg_df["ma28"] = agg_df["sales_qty"].rolling(28, min_periods=1).mean()

        # Build subplot: sales on row 1, optional promo flags on row 2
        has_promo = "is_promotion" in plot_df.columns
        if show_promo and has_promo:
            promo_agg = (
                plot_df.groupby("date")["is_promotion"].max().reset_index()
            )
            fig_trends = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.80, 0.20],
                vertical_spacing=0.04,
                subplot_titles=("Daily Sales", "Promotion Active"),
            )
            row_sales = 1
        else:
            fig_trends = make_subplots(rows=1, cols=1)
            row_sales = 1

        # Sales line
        fig_trends.add_trace(
            go.Scatter(
                x=agg_df["date"],
                y=agg_df["sales_qty"],
                mode="lines",
                name="Sales",
                line={"color": "royalblue", "width": 1.5},
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Sales: %{y:,.0f}<extra></extra>",
            ),
            row=row_sales, col=1,
        )

        if show_ma:
            fig_trends.add_trace(
                go.Scatter(
                    x=agg_df["date"],
                    y=agg_df["ma7"],
                    mode="lines",
                    name="7-day MA",
                    line={"color": "orange", "width": 1.8, "dash": "dot"},
                    hovertemplate="7d MA: %{y:,.0f}<extra></extra>",
                ),
                row=row_sales, col=1,
            )
            fig_trends.add_trace(
                go.Scatter(
                    x=agg_df["date"],
                    y=agg_df["ma28"],
                    mode="lines",
                    name="28-day MA",
                    line={"color": "firebrick", "width": 1.8, "dash": "dash"},
                    hovertemplate="28d MA: %{y:,.0f}<extra></extra>",
                ),
                row=row_sales, col=1,
            )

        if show_promo and has_promo:
            fig_trends.add_trace(
                go.Bar(
                    x=promo_agg["date"],
                    y=promo_agg["is_promotion"],
                    name="Promo",
                    marker_color="rgba(50,200,100,0.5)",
                    hovertemplate="%{x|%d %b %Y}: Promo=%{y}<extra></extra>",
                ),
                row=2, col=1,
            )

        fig_trends.update_layout(
            height=480 if (show_promo and has_promo) else 380,
            template="plotly_white",
            hovermode="x unified",
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 50, "r": 30, "t": 40, "b": 60},
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(label="All", step="all"),
                    ]
                ),
                rangeslider=dict(visible=True, thickness=0.06),
                type="date",
            ),
        )
        st.plotly_chart(fig_trends, use_container_width=True, config=_PLOTLY_CONFIG)

        # KPI strip below the chart
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Sales", f"{agg_df['sales_qty'].sum():,.0f}")
        k2.metric("Daily Average", f"{agg_df['sales_qty'].mean():,.0f}")
        k3.metric("Peak Day", f"{agg_df['sales_qty'].max():,.0f}")
        k4.metric("Min Day", f"{agg_df['sales_qty'].min():,.0f}")

    # ── Tab 2: Forecast ───────────────────────────────────────────────────
    with tab_forecast:
        st.subheader(f"42-Day Forecast — {best_name}")

        fc_col1, fc_col2 = st.columns([3, 1])
        with fc_col2:
            history_window = st.slider(
                "History window (days)", min_value=30, max_value=365,
                value=90, step=15,
                help="How many historical days to show alongside the forecast.",
            )
            show_ci = st.checkbox("Show confidence interval", value=True)

        hist_agg = df_features.groupby("date")["sales_qty"].sum().reset_index()
        hist_agg = hist_agg.sort_values("date")
        hist_recent = hist_agg.tail(history_window)

        fig_fc = go.Figure()

        # Confidence band
        if show_ci:
            fig_fc.add_trace(
                go.Scatter(
                    x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
                    y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(220,50,50,0.12)",
                    line={"color": "rgba(0,0,0,0)"},
                    name="Confidence Interval",
                    hoverinfo="skip",
                )
            )

        # Historical line
        fig_fc.add_trace(
            go.Scatter(
                x=hist_recent["date"],
                y=hist_recent["sales_qty"],
                mode="lines",
                name="Historical",
                line={"color": "royalblue", "width": 2},
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
            )
        )

        # Forecast line
        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_sales"],
                mode="lines+markers",
                name=f"{best_name} Forecast",
                line={"color": "crimson", "dash": "dash", "width": 2},
                marker={"size": 5, "color": "crimson"},
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Forecast: %{y:,.0f}<br>"
                    f"Lower: %{{customdata[0]:,.0f}}<br>"
                    f"Upper: %{{customdata[1]:,.0f}}"
                    "<extra></extra>"
                ),
                customdata=np.stack(
                    [forecast_df["lower_bound"].values, forecast_df["upper_bound"].values], axis=1
                ),
            )
        )

        # Vertical split line where history ends / forecast begins
        split_date = hist_recent["date"].max()
        fig_fc.add_shape(
            type="line",
            x0=split_date, x1=split_date,
            y0=0, y1=1,
            xref="x", yref="paper",
            line={"dash": "dot", "color": "grey", "width": 1.5},
        )
        fig_fc.add_annotation(
            x=split_date, y=1,
            xref="x", yref="paper",
            text="Forecast start",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": "grey"},
        )

        # Annotate peak forecast day
        peak_idx = forecast_df["predicted_sales"].idxmax()
        peak_row = forecast_df.loc[peak_idx]
        fig_fc.add_annotation(
            x=peak_row["date"],
            y=peak_row["predicted_sales"],
            text=f"Peak: {peak_row['predicted_sales']:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="crimson",
            font={"color": "crimson", "size": 11},
            bgcolor="rgba(255,255,255,0.8)",
        )

        fig_fc.update_layout(
            height=420,
            template="plotly_white",
            hovermode="x unified",
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 50, "r": 30, "t": 40, "b": 60},
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.06),
                type="date",
            ),
            yaxis_title="Sales Quantity",
        )
        with fc_col1:
            st.plotly_chart(fig_fc, use_container_width=True, config=_PLOTLY_CONFIG)

        # Forecast KPIs
        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("Total Forecast (42d)", f"{forecast_df['predicted_sales'].sum():,.0f}")
        fk2.metric("Daily Avg Forecast", f"{forecast_df['predicted_sales'].mean():,.0f}")
        fk3.metric("Peak Forecast Day", f"{forecast_df['predicted_sales'].max():,.0f}")
        fk4.metric(
            "vs Last 42d Actual",
            f"{forecast_df['predicted_sales'].sum():,.0f}",
            delta=f"{forecast_df['predicted_sales'].sum() - hist_agg.tail(42)['sales_qty'].sum():+,.0f}",
        )

        # Forecast table with colour gradient
        st.subheader("Forecast Table")
        display_fc = forecast_df[["date", "predicted_sales", "lower_bound", "upper_bound"]].copy()
        display_fc["predicted_sales"] = display_fc["predicted_sales"].round(0).astype(int)
        display_fc["lower_bound"] = display_fc["lower_bound"].round(0).astype(int)
        display_fc["upper_bound"] = display_fc["upper_bound"].round(0).astype(int)
        display_fc.columns = ["Date", "Predicted Sales", "Lower Bound", "Upper Bound"]
        st.dataframe(
            display_fc.style.background_gradient(subset=["Predicted Sales"], cmap="RdYlGn"),
            use_container_width=True,
            height=320,
        )

        # Download button
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Forecast CSV",
            data=csv_bytes,
            file_name=f"forecast_{st.session_state['uploaded_filename']}",
            mime="text/csv",
        )

    # ── Tab 3: Model Comparison ──────────────────────────────────────────
    with tab_compare:
        st.subheader("Model Performance Comparison")

        results = {"Prophet": prophet_metrics, "XGBoost": xgb_metrics}

        # ── Grouped bar: MAE / RMSE / MAPE ───────────────────────────────
        metrics_names = ["MAE", "RMSE", "MAPE"]
        colors = {"MAE": "steelblue", "RMSE": "mediumseagreen", "MAPE": "tomato"}

        fig_bar = go.Figure()
        for metric in metrics_names:
            fig_bar.add_trace(
                go.Bar(
                    name=metric,
                    x=["Prophet", "XGBoost"],
                    y=[results["Prophet"][metric], results["XGBoost"][metric]],
                    marker_color=colors[metric],
                    text=[f"{results['Prophet'][metric]:.2f}", f"{results['XGBoost'][metric]:.2f}"],
                    textposition="outside",
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
                )
            )

        fig_bar.update_layout(
            barmode="group",
            template="plotly_white",
            height=340,
            title="All Metrics — Side by Side",
            yaxis_title="Score",
            legend={"orientation": "h", "y": -0.2},
            margin={"t": 50, "b": 60},
        )

        # ── Radar chart ───────────────────────────────────────────────────
        # Normalise each metric to [0,1] (lower is better → invert)
        def _norm_invert(val, all_vals):
            mn, mx = min(all_vals), max(all_vals)
            if mx == mn:
                return 0.5
            return 1 - (val - mn) / (mx - mn)   # higher score = better

        radar_metrics = ["MAE", "RMSE", "MAPE"]
        prophet_scores = [
            _norm_invert(prophet_metrics[m], [prophet_metrics[m], xgb_metrics[m]])
            for m in radar_metrics
        ]
        xgb_scores = [
            _norm_invert(xgb_metrics[m], [prophet_metrics[m], xgb_metrics[m]])
            for m in radar_metrics
        ]
        # Close the polygon
        radar_cats = radar_metrics + [radar_metrics[0]]
        prophet_scores_closed = prophet_scores + [prophet_scores[0]]
        xgb_scores_closed = xgb_scores + [xgb_scores[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=prophet_scores_closed,
            theta=radar_cats,
            fill="toself",
            name="Prophet",
            line_color="royalblue",
            fillcolor="rgba(65,105,225,0.2)",
            hovertemplate="Prophet — %{theta}: %{r:.2f}<extra></extra>",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=xgb_scores_closed,
            theta=radar_cats,
            fill="toself",
            name="XGBoost",
            line_color="tomato",
            fillcolor="rgba(255,99,71,0.2)",
            hovertemplate="XGBoost — %{theta}: %{r:.2f}<extra></extra>",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white",
            height=340,
            title="Normalised Score (higher = better)",
            legend={"orientation": "h", "y": -0.15},
            margin={"t": 60, "b": 60},
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(fig_bar, use_container_width=True, config=_PLOTLY_CONFIG)
        with chart_col2:
            st.plotly_chart(fig_radar, use_container_width=True, config=_PLOTLY_CONFIG)

        # ── Styled metrics table ──────────────────────────────────────────
        comparison_data = [
            {
                "Model": "Prophet",
                "MAE": prophet_metrics["MAE"],
                "RMSE": prophet_metrics["RMSE"],
                "MAPE (%)": prophet_metrics["MAPE"],
                "Winner": "✅" if best_name == "Prophet" else "",
            },
            {
                "Model": "XGBoost",
                "MAE": xgb_metrics["MAE"],
                "RMSE": xgb_metrics["RMSE"],
                "MAPE (%)": xgb_metrics["MAPE"],
                "Winner": "✅" if best_name == "XGBoost" else "",
            },
        ]
        compare_df = pd.DataFrame(comparison_data)

        def _highlight_winner(row):
            if row["Winner"] == "✅":
                return ["background-color: #d4edda; color: #155724"] * len(row)
            return [""] * len(row)

        styled = compare_df.style.apply(_highlight_winner, axis=1).format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}"}
        )
        st.dataframe(styled, use_container_width=True)
        st.caption(
            "Lower MAPE = better accuracy. Under 15% is good. Under 10% is excellent."
        )
