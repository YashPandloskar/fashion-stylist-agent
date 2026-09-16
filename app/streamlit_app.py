"""
Streamlit frontend for the Fashion Stylist Agent.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Fashion Stylist",
    page_icon="👗",
    layout="wide",
)

st.title("👗 AI Fashion Stylist Agent")
st.caption(
    "Powered by LangGraph · FashionCLIP RAG · Generative AI"
)

# ── Input form ─────────────────────────────────────────────────────────────────

with st.form("outfit_form"):
    col1, col2 = st.columns(2)

    with col1:
        occasion = st.selectbox(
            "Occasion",
            ["Business meeting", "Casual day out", "Evening dinner", "First date",
             "Wedding guest", "Job interview", "Festival", "Gym / sport", "Travel"],
        )
        gender = st.selectbox("Gender", ["Women", "Men", "Unisex"])
        style_preference = st.text_input(
            "Style preference (optional)",
            placeholder="e.g. minimalist, streetwear, smart casual",
        )

    with col2:
        time_of_day = st.selectbox(
            "Time of day", ["Morning", "Afternoon", "Evening", "Night"]
        )
        weather = st.selectbox(
            "Weather",
            ["Sunny and warm", "Hot", "Cool / mild", "Cold", "Rainy", "Snowy"],
        )

    submitted = st.form_submit_button("✨ Generate Outfit", use_container_width=True)

# ── Agent call & result display ────────────────────────────────────────────────

if submitted:
    with st.spinner("Styling your outfit — this takes 15–30 seconds..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={
                    "occasion": occasion,
                    "gender": gender,
                    "time_of_day": time_of_day,
                    "weather": weather,
                    "style_preference": style_preference or "smart casual",
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    if data.get("error"):
        st.warning(f"Agent encountered an issue: {data['error']}")

    # ── Layout: image left, outfit details right ──────────────────────────────
    img_col, detail_col = st.columns([1, 1])

    with img_col:
        if data.get("image_base64"):
            img_bytes = base64.b64decode(data["image_base64"])
            image = Image.open(io.BytesIO(img_bytes))
            st.image(image, use_column_width=True)
        else:
            st.info("No image generated.")

    with detail_col:
        # Stylist narration
        st.subheader("Stylist's Note")
        st.write(data.get("stylist_narration", ""))

        # Outfit pieces
        st.subheader("The Outfit")
        outfit = data.get("outfit", {})
        for slot, details in outfit.items():
            if isinstance(details, dict):
                with st.expander(f"**{slot.capitalize()}** — {details.get('name', '')}"):
                    st.write(details.get("reason", ""))

        # Style brief (collapsible — useful for demos)
        with st.expander("View style brief (agent reasoning step 1)"):
            st.write(data.get("style_brief", ""))
