import streamlit as st
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization
import explainer
import streamlit.components.v1 as components
import importlib

# Page config
st.set_page_config(page_title="Explainable Opinion Mining", layout="wide")

@st.cache_resource
def load_explainer_v3():

    # Force reload of explainer module to pick up changes
    importlib.reload(explainer)

    # Use the final IMDb-trained model
    model_path = "src/results_imdb"
    
    import os
    if not os.path.exists(model_path):
        # Fail gracefully if the expected model is missing
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
        
    return explainer.ModelExplainer(model_path=model_path, device=torch.device("cpu"))

def main():
    st.title("🔍 Explainable Opinion Mining")
    st.markdown("Analyze sentiment and understand *why* the model made its prediction using SHAP and Integrated Gradients.")

    # User Input
    text = st.text_area("Enter text to analyze:", value="I absolutely loved this movie! The acting was superb.")

    if st.button("Analyze"):
        if not text:
            st.warning("Please enter some text.")
            return

        with st.spinner("Loading model... (this may take a moment)"):
            explainer = load_explainer_v3()
        st.success(f"Model loaded on {explainer.device}")
        
        # Get prediction first to determine labels and default
        try:
            # We can use the pipeline directly
            preds = explainer.pipeline(text)[0] # List of dicts [{'label': 'LABEL_0', 'score': ...}, ...]
            # Sort by label ID to ensure order
            preds = sorted(preds, key=lambda x: int(x['label'].split('_')[-1]) if '_' in x['label'] else 0)
            
            num_labels = len(preds)
            if num_labels == 3:
                label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            else:
                label_map = {0: "Negative", 1: "Positive"}
                
            # Find predicted class
            probs = [p['score'] for p in preds]
            pred_idx = int(np.argmax(probs))
            pred_label = label_map.get(pred_idx, f"Label {pred_idx}")
            
            st.info(f"Prediction: **{pred_label}** (Confidence: {probs[pred_idx]:.2f})")
            
            # Class Selection for Explanation
            selected_label_name = st.selectbox("Explain for Class:", list(label_map.values()), index=pred_idx)
            # Find index of selected name
            target_class_idx = [k for k, v in label_map.items() if v == selected_label_name][0]

            st.markdown("---")
            
            # Tabs for explanations
            tab1, tab2 = st.tabs(["SHAP", "Integrated Gradients"])

            with tab1:
                st.header("SHAP (Shapley Additive exPlanations)")
                with st.spinner("Calculating SHAP values..."):
                    shap_values = explainer.get_shap_values(text)
                    
                    # SHAP Text Plot
                    st.subheader("Text Attribution Plot")
                    try:
                        # target_class_idx gives us the explanation object for the selected class
                        shap_explanation = shap_values[0, :, target_class_idx]
                        
                        # shap.plots.text handles it.
                        shap_html = shap.plots.text(shap_explanation, display=False)
                        
                        # Wrap in a white container
                        styled_shap_html = f"""
                        <div style="background-color: white; color: black; padding: 20px; border-radius: 10px;">
                            {shap_html}
                        </div>
                        """
                        components.html(styled_shap_html, height=450, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"Could not render SHAP plot: {e}")

            with tab2:
                st.header("Integrated Gradients (Captum)")
                with st.spinner("Calculating Integrated Gradients..."):
                    # Pass target class index to explain the selected class
                    vis_record = explainer.get_integrated_gradients(text, target=target_class_idx)
                    
                    html_obj = visualization.visualize_text([vis_record])
                    styled_captum_html = f"<div style='background-color: white; color: black; padding: 10px; border-radius: 5px;'>{html_obj.data}</div>"
                    components.html(styled_captum_html, height=600, scrolling=True)

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.write("Please check if the model is correctly trained or try again.")

if __name__ == "__main__":
    main()
