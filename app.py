
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = Path(__file__).with_name("model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Explore how selected passenger details relate to the model's "
    "estimated survival probability."
)

st.subheader("Enter passenger details")
pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=2)
sex = st.radio("Sex", options=["female", "male"], horizontal=True)
age = st.slider("Age (years)", min_value=1, max_value=80, value=30)
fare = st.slider("Fare paid", min_value=0.0, max_value=512.0, value=32.0, step=1.0)

if st.button("Predict survival probability", type="primary", use_container_width=True):
    try:
        model = load_model()
        input_row = pd.DataFrame(
            [[int(pclass), sex, float(age), float(fare)]],
            columns=FEATURES
        )

        if input_row.isnull().any().any():
            st.error("Please complete all passenger inputs.")
        else:
            probability = float(model.predict_proba(input_row)[0][1])
            predicted_class = int(model.predict(input_row)[0])

            st.metric("Estimated survival probability", f"{probability:.1%}")
            if predicted_class == 1:
                st.success("The model predicts the 'survived' class.")
            else:
                st.info("The model predicts the 'did not survive' class.")

    except FileNotFoundError:
        st.error("The model file is missing. Keep model.joblib in the same folder as app.py.")
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Please check the input values. Details: {error}")
    except Exception:
        st.error("Something went wrong while making the prediction. Check that model.joblib matches this app.")

st.caption(
    "Model: Logistic Regression trained on the public historical Titanic passenger dataset. "
    "This is an educational example, not advice or a reliable prediction about any individual."
)
