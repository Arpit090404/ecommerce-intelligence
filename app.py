import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide"
)

# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    delivery_model     = joblib.load('models/delivery_model.pkl')
    delivery_explainer = joblib.load('models/delivery_explainer.pkl')
    kmeans_model       = joblib.load('models/kmeans_model.pkl')
    kmeans_scaler      = joblib.load('models/kmeans_scaler.pkl')
    return delivery_model, delivery_explainer, kmeans_model, kmeans_scaler

@st.cache_data
def load_data():
    rfm = pd.read_csv('data/rfm_segments.csv')
    return rfm

delivery_model, delivery_explainer, kmeans_model, kmeans_scaler = load_models()
rfm = load_data()

segment_map = {1: 'High Value', 3: 'Champions', 0: 'Casual', 2: 'Lost'}
segment_colors = {
    'Champions':  '#2ECC71',
    'High Value': '#3498DB',
    'Casual':     '#E74C3C',
    'Lost':       '#95A5A6'
}

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🛒 E-Commerce Intelligence Platform")
st.markdown("*Olist Brazilian E-Commerce — Customer & Delivery Analytics*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Customer Lookup", "Segment Overview", "Delivery Predictor"])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Customer Lookup
# ════════════════════════════════════════════════════════════════════════════
if page == "Customer Lookup":
    st.header("Customer Lookup")

    customer_id = st.text_input(
        "Enter Customer Unique ID",
        placeholder="e.g. 0000366f3b9a7992bf8c76cfdf3221e2"
    )

    if customer_id:
        customer = rfm[rfm['customer_id'] == customer_id]

        if customer.empty:
            st.error("Customer not found. Try another ID.")
        else:
            row = customer.iloc[0]
            segment = row['segment']
            color = segment_colors.get(segment, '#333')

            # ── KPI cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segment", segment)
            col2.metric("Recency", f"{int(row['recency_days'])} days")
            col3.metric("Frequency", f"{int(row['frequency'])} orders")
            col4.metric("Monetary", f"R$ {row['monetary']:.2f}")

            st.divider()

            # ── Segment badge
            st.markdown(
                f"<div style='background:{color};padding:12px 20px;"
                f"border-radius:8px;color:white;font-weight:600;"
                f"font-size:18px;display:inline-block'>"
                f"🏷️ {segment}</div>",
                unsafe_allow_html=True
            )

            # ── Recommended action
            st.subheader("Recommended Action")
            actions = {
                'Champions':  "✅ Reward them — send loyalty perks or early access to new products.",
                'High Value': "💎 Nurture them — offer premium support and exclusive discounts.",
                'Casual':     "📢 Engage them — send re-engagement email with a discount code.",
                'Lost':       "⚠️ Win-back campaign — offer a significant discount to re-activate."
            }
            st.info(actions.get(segment, "No action available."))

    else:
        # show sample IDs to try
        st.markdown("**Sample customer IDs to try:**")
        st.dataframe(rfm[['customer_id', 'segment', 'recency_days', 'monetary']].head(10))

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Segment Overview
# ════════════════════════════════════════════════════════════════════════════
elif page == "Segment Overview":
    st.header("Customer Segment Overview")

    summary = rfm.groupby('segment').agg(
        Customers=('customer_id', 'count'),
        Avg_Recency=('recency_days', 'mean'),
        Avg_Monetary=('monetary', 'mean')
    ).round(2).sort_values('Avg_Monetary', ascending=False)

    st.dataframe(summary, use_container_width=True)
    st.divider()

    # segment distribution bar chart
    st.subheader("Segment Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [segment_colors.get(s, '#333') for s in summary.index]
    ax.bar(summary.index, summary['Customers'], color=colors)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Customers per Segment')
    st.pyplot(fig)

    # monetary comparison
    st.subheader("Average Spend per Segment")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(summary.index, summary['Avg_Monetary'], color=colors)
    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Avg Monetary (R$)')
    ax2.set_title('Average Spend per Segment')
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Delivery Predictor
# ════════════════════════════════════════════════════════════════════════════
elif page == "Delivery Predictor":
    st.header("Delivery Delay Predictor")
    st.markdown("Predict how many days early/late an order will arrive, with SHAP explanation.")

    col1, col2 = st.columns(2)

    with col1:
        estimated_days   = st.slider("Estimated delivery days", 1, 60, 15)
        freight_value    = st.number_input("Freight value (R$)", 0.0, 500.0, 20.0)
        price            = st.number_input("Product price (R$)", 0.0, 5000.0, 100.0)
        product_weight   = st.number_input("Product weight (g)", 0, 30000, 500)

    with col2:
        product_length   = st.number_input("Product length (cm)", 0, 100, 30)
        product_height   = st.number_input("Product height (cm)", 0, 100, 15)
        product_width    = st.number_input("Product width (cm)", 0, 100, 20)
        order_month      = st.selectbox("Order month", range(1, 13))
        order_dayofweek  = st.selectbox("Day of week (1=Mon, 7=Sun)", range(1, 8))
        product_category = st.number_input("Product category (encoded)", 0, 100, 10)
        seller_state     = st.number_input("Seller state (encoded)", 0, 30, 5)
        customer_state   = st.number_input("Customer state (encoded)", 0, 30, 10)

    if st.button("Predict Delay", type="primary"):
        input_data = pd.DataFrame([{
            'estimated_delivery_days': estimated_days,
            'freight_value':           freight_value,
            'price':                   price,
            'product_weight_g':        product_weight,
            'product_length_cm':       product_length,
            'product_height_cm':       product_height,
            'product_width_cm':        product_width,
            'order_month':             order_month,
            'order_dayofweek':         order_dayofweek,
            'product_category_name':   product_category,
            'seller_state':            seller_state,
            'customer_state':          customer_state
        }])

        prediction = delivery_model.predict(input_data)[0]

        if prediction < 0:
            st.success(f"📦 Predicted: **{abs(prediction):.1f} days EARLY**")
        elif prediction == 0:
            st.info("📦 Predicted: **On time**")
        else:
            st.warning(f"⚠️ Predicted: **{prediction:.1f} days LATE**")

        # SHAP force plot
        st.subheader("Why this prediction?")
        shap_vals = delivery_explainer.shap_values(input_data)
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            delivery_explainer.expected_value,
            shap_vals[0],
            input_data.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig3)