import streamlit as st
import requests
import os

def main():
    st.set_page_config(
        page_title="Car Damage Estimation",
        page_icon=":car:",
        layout="wide"
    )

    # Title and Description
    st.title("🚗 Car Damage Estimation AI")
    st.write("Upload a car image to get an instant damage assessment")

    # Backend URL configuration
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

    # Sidebar for additional inputs
    st.sidebar.header("Vehicle Details (Optional)")
    car_name = st.sidebar.text_input("Car Name (e.g., Toyota Camry)")
    car_model = st.sidebar.text_input("Car Model Year (e.g., 2022)")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Car Image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the car showing the damage"
    )

    # Estimation button
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Car Image", use_container_width=True)
        
        with col2:
            if st.button("Estimate Damage", type="primary"):
                with st.spinner('Analyzing image...'):
                    try:
                        # Prepare files and parameters for upload
                        files = {
                            'file': (uploaded_file.name, uploaded_file, uploaded_file.type)
                        }
                        
                        # Prepare query parameters
                        params = {}
                        if car_name:
                            params['car_name'] = car_name
                        if car_model:
                            params['car_model'] = car_model

                        # Send request to backend
                        response = requests.post(
                            f"{BACKEND_URL}/estimate-damage", 
                            files=files,
                            params=params
                        )

                        # Check response
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results in a structured manner
                            st.subheader("Damage Estimation Results")
                            
                            # Create columns for better layout
                            res_col1, res_col2 = st.columns(2)
                            
                            with res_col1:
                                st.markdown(f"{result}")
                                # st.write(f"**Model:** {result.get('model', 'Unknown')}")
                                # st.write(f"**Car Price:** {result.get('car_price', 'N/A')}")
                            
                            # with res_col2:
                            #     st.write(f"**Damage Description:** {result.get('damage_description', 'No damage detected')}")
                            #     st.write(f"**Damage Estimation:** {result.get('damage_estimation', 'Unknown')}")
                            #     st.write(f"**Total Estimated Price:** {result.get('total_estimated_price', 'N/A')}")
                            
                            # # Damage severity visualization
                            # damage_description = result.get('damage_description', '').lower()
                            # if 'minor' in damage_description:
                            #     severity = "🟢 Minor"
                            # elif 'moderate' in damage_description:
                            #     severity = "🟠 Moderate"
                            # elif 'severe' in damage_description:
                            #     severity = "🔴 Severe"
                            # else:
                            #     severity = "🟢 No Significant Damage"
                            
                            # st.warning(f"**Damage Severity:** {severity}")

                        else:
                            st.error(f"Error: {response.text}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    # Footer
    st.markdown("---")
    st.markdown("🤖 Powered by NVIDIA LLaMA Vision AI")

if __name__ == "__main__":
    main()