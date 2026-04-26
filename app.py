import streamlit as st
import torch
import numpy as np
import os
import plotly.graph_objects as go
from models.lstm_yield import LSTMYieldModel
from advisory import get_yield_advisory

def get_task_feedback(action_text):
    """
    Provide educational feedback explaining why a task is important.
    Returns a short, farmer-friendly explanation.
    """
    action_lower = action_text.lower()
    
    # Irrigation-related tasks
    if "irrigation" in action_lower or "water" in action_lower:
        if "apply" in action_lower or "within" in action_lower:
            return "💧 **Why this matters:** Regular irrigation prevents moisture stress that can reduce yield by up to 30%. Water stress during critical growth stages affects final harvest."
        else:
            return "💧 **Why this matters:** Consistent soil moisture helps your crop grow steadily. Too little water causes stress, while too much can lead to root problems."
    
    # Leaf checking tasks
    elif "leaf" in action_lower or "leaves" in action_lower:
        if "yellowing" in action_lower:
            return "🍃 **Why this matters:** Yellow leaves can signal nutrient deficiency, disease, or water stress. Early detection lets you fix problems before they spread and reduce yield."
        else:
            return "🍃 **Why this matters:** Checking leaves helps catch diseases and nutrient problems early. Early treatment saves your crop and protects your harvest."
    
    # Growth monitoring tasks
    elif "monitor" in action_lower or "growth" in action_lower or "watch" in action_lower:
        if "daily" in action_lower:
            return "👀 **Why this matters:** Daily monitoring helps you spot problems immediately. Early season issues, if caught quickly, are easier to fix and won't hurt your final yield."
        else:
            return "👀 **Why this matters:** Regular monitoring helps you catch problems early. Early detection means easier fixes and better harvest results."
    
    # Soil moisture tasks
    elif "soil moisture" in action_lower or "moisture" in action_lower:
        return "🌱 **Why this matters:** Proper soil moisture is essential for healthy roots. Too dry or too wet soil can stress your crop and reduce growth."
    
    # Fertilizer tasks
    elif "fertilizer" in action_lower or "nitrogen" in action_lower:
        if "reduce" in action_lower:
            return "🌾 **Why this matters:** Too much nitrogen can burn leaves and waste money. Balanced fertilizer use protects your crop and saves costs."
        else:
            return "🌾 **Why this matters:** Timely fertilizer application supports healthy growth. Applying at the right time maximizes nutrient uptake and improves yield."
    
    # Drainage tasks
    elif "drainage" in action_lower or "waterlogged" in action_lower:
        return "🚰 **Why this matters:** Good drainage prevents root rot and waterlogging. Standing water can suffocate roots and reduce crop health."
    
    # Disease/pest checking tasks
    elif "disease" in action_lower or "pest" in action_lower or "root rot" in action_lower:
        return "🦠 **Why this matters:** Early disease detection prevents spread to healthy plants. Quick action protects your entire field and saves your harvest."
    
    # Water level tasks (rice-specific)
    elif "water level" in action_lower or "paddy" in action_lower:
        return "🌾 **Why this matters:** Proper water levels in paddy fields are critical for rice. Too little water stresses plants, while too much can cause problems."
    
    # Seed germination tasks
    elif "germination" in action_lower or "seed" in action_lower:
        return "🌱 **Why this matters:** Good germination conditions ensure strong, healthy seedlings. Weak starts can lead to poor growth and lower yields."
    
    # Stress detection tasks
    elif "stress" in action_lower or "signs" in action_lower:
        return "⚠️ **Why this matters:** Early stress detection lets you take action before problems worsen. Addressing stress early protects your yield potential."
    
    # Default feedback
    return "✅ **Why this matters:** This task helps maintain healthy crop growth and protects your harvest potential."

def get_weather_description(crop, agro_climatic_zone):
    """
    Generate descriptive weather information based on crop and agro-climatic zone.
    Base weather comes from agro-climatic zone, contextualized for specific crop needs.
    Rule-based only - no APIs or real-time data.
    Does not affect model predictions.
    """
    if not agro_climatic_zone:
        return "Weather information is not available for your region."
    
    if agro_climatic_zone == "Coastal":
        if crop == "Rice":
            return "Warm temperatures with high humidity throughout the season. The high humidity and water availability are well-suited for rice cultivation. Occasional rainfall provides good moisture for paddy fields. Sea breeze helps moderate extreme heat."
        elif crop == "Wheat":
            return "Warm temperatures with high humidity during the growing season. Wheat needs balanced moisture, so the high humidity requires careful monitoring to prevent excess moisture issues. Occasional rainfall may require careful drainage management. Coastal winds help reduce extreme temperatures."
        else:  # Maize
            return "Warm temperatures with high humidity levels. Maize benefits from the moisture but requires good drainage to prevent waterlogging. Occasional rainfall supports growth. Coastal conditions provide moderate temperature variations."
    
    elif agro_climatic_zone == "Semi-arid":
        if crop == "Rice":
            return "Hot days with intense sunlight and low rainfall. Rice requires consistent water, so the dry conditions need careful water management and irrigation for paddy cultivation. Temperature variations between day and night are significant."
        elif crop == "Wheat":
            return "Hot days with low rainfall and dry conditions. Wheat needs balanced moisture, so irrigation is essential to maintain optimal growing conditions. Cooler nights during winter months support wheat growth. Water stress is common and requires careful management."
        else:  # Maize
            return "Hot days with low rainfall and dry conditions. Maize requires adequate water during critical growth stages. High temperature stress during peak summer can affect development. Irrigation is critical for successful maize cultivation in this zone."
    
    elif agro_climatic_zone == "Tropical":
        if crop == "Rice":
            return "Moderate temperatures with regular rainfall throughout the season. The high humidity and consistent water availability are ideal for rice cultivation. High humidity supports paddy growth. Consistent moisture levels are favorable for rice."
        elif crop == "Wheat":
            return "Moderate temperatures with regular rainfall patterns. Wheat needs balanced moisture, so monitor humidity levels to avoid excess moisture that can affect grain quality. Balanced conditions support wheat growth. Adequate moisture and temperature control are generally favorable."
        else:  # Maize
            return "Moderate temperatures with regular rainfall. Maize benefits from the balanced moisture conditions. Temperature and humidity levels are generally favorable for maize growth. Adequate moisture supports healthy development."
    
    return "Weather patterns vary by season and location."

def get_weather_risks(crop, agro_climatic_zone):
    """
    Determine possible weather risks based on crop type and agro-climatic zone.
    Risks are prioritized based on crop-specific vulnerabilities.
    Rule-based only - no numeric values or probabilities.
    Returns a list of risk indicators with emojis and short text.
    Does not affect ML logic.
    """
    if not agro_climatic_zone:
        return []
    
    risks = []
    
    # Rice: prioritize excess rain and humidity-related disease risks
    if crop == "Rice":
        if agro_climatic_zone == "Coastal":
            risks.append("🌧️ Excess rain risk")
            risks.append("🌫️ Humidity-related disease risk")
        elif agro_climatic_zone == "Semi-arid":
            risks.append("💧 Water stress risk")
            risks.append("🌫️ Low humidity stress risk")
        elif agro_climatic_zone == "Tropical":
            risks.append("🌧️ Excess rain risk")
            risks.append("🌫️ Humidity-related disease risk")
    
    # Wheat: prioritize humidity and heat stress risks
    elif crop == "Wheat":
        if agro_climatic_zone == "Coastal":
            risks.append("🌫️ High humidity risk")
            risks.append("☀️ Heat stress possible")
        elif agro_climatic_zone == "Semi-arid":
            risks.append("☀️ Heat stress possible")
            risks.append("💧 Water stress risk")
        elif agro_climatic_zone == "Tropical":
            risks.append("🌫️ High humidity risk")
            risks.append("☀️ Heat stress possible")
    
    # Maize: prioritize waterlogging and leaf disease risks
    else:  # Maize
        if agro_climatic_zone == "Coastal":
            risks.append("🌧️ Waterlogging risk")
            risks.append("🍃 Leaf disease risk")
        elif agro_climatic_zone == "Semi-arid":
            risks.append("☀️ Heat stress possible")
            risks.append("💧 Water stress risk")
        elif agro_climatic_zone == "Tropical":
            risks.append("🌧️ Waterlogging risk")
            risks.append("🍃 Leaf disease risk")
    
    return risks

def get_conditional_weather_advice(crop, agro_climatic_zone):
    """
    Generate conditional weather advice based on crop type and agro-climatic zone.
    Advice is tailored to crop-specific vulnerabilities.
    Returns a list of tuples: (condition_emoji, condition_text, actions_list)
    Rule-based only - does not affect ML logic or predictions.
    """
    if not agro_climatic_zone:
        return []
    
    advice = []
    
    # Rice: prioritize excess rain and humidity-related disease risks
    if crop == "Rice":
        if agro_climatic_zone == "Coastal":
            advice.append((
                "🌧️",
                "If heavy rain occurs:",
                [
                    "Check paddy field water levels",
                    "Ensure proper field drainage",
                    "Avoid fertilizer application during heavy rain"
                ]
            ))
            advice.append((
                "🌫️",
                "If humidity stays very high:",
                [
                    "Monitor for rice blast and bacterial leaf blight",
                    "Ensure good air circulation in fields",
                    "Avoid overhead irrigation during high humidity"
                ]
            ))
        elif agro_climatic_zone == "Semi-arid":
            advice.append((
                "💧",
                "If water becomes scarce:",
                [
                    "Maintain consistent paddy field water levels",
                    "Use efficient irrigation methods",
                    "Monitor for water stress in rice plants"
                ]
            ))
            advice.append((
                "☀️",
                "If very hot days continue:",
                [
                    "Increase irrigation frequency",
                    "Provide shade if possible",
                    "Monitor for heat stress symptoms"
                ]
            ))
        elif agro_climatic_zone == "Tropical":
            advice.append((
                "🌧️",
                "If heavy rain occurs:",
                [
                    "Check paddy field water levels",
                    "Ensure proper field drainage",
                    "Avoid fertilizer application during heavy rain"
                ]
            ))
            advice.append((
                "🌫️",
                "If humidity stays very high:",
                [
                    "Monitor for rice blast and bacterial leaf blight",
                    "Ensure good air circulation in fields",
                    "Avoid overhead irrigation during high humidity"
                ]
            ))
    
    # Wheat: prioritize humidity and heat stress risks
    elif crop == "Wheat":
        if agro_climatic_zone == "Coastal":
            advice.append((
                "🌫️",
                "If humidity stays very high:",
                [
                    "Monitor for rust and powdery mildew diseases",
                    "Ensure good air circulation between plants",
                    "Avoid overhead irrigation during high humidity"
                ]
            ))
            advice.append((
                "☀️",
                "If very hot days continue:",
                [
                    "Increase irrigation frequency",
                    "Avoid midday spraying",
                    "Monitor for heat stress during grain filling"
                ]
            ))
        elif agro_climatic_zone == "Semi-arid":
            advice.append((
                "☀️",
                "If very hot days continue:",
                [
                    "Increase irrigation frequency",
                    "Avoid midday spraying",
                    "Monitor for heat stress during grain filling"
                ]
            ))
            advice.append((
                "💧",
                "If water becomes scarce:",
                [
                    "Prioritize critical growth stages",
                    "Use drip irrigation",
                    "Reduce water loss with mulching"
                ]
            ))
        elif agro_climatic_zone == "Tropical":
            advice.append((
                "🌫️",
                "If humidity stays very high:",
                [
                    "Monitor for rust and powdery mildew diseases",
                    "Ensure good air circulation between plants",
                    "Avoid overhead irrigation during high humidity"
                ]
            ))
            advice.append((
                "☀️",
                "If very hot days continue:",
                [
                    "Increase irrigation frequency",
                    "Avoid midday spraying",
                    "Monitor for heat stress during grain filling"
                ]
            ))
    
    # Maize: prioritize waterlogging and leaf disease risks
    else:  # Maize
        if agro_climatic_zone == "Coastal":
            advice.append((
                "🌧️",
                "If heavy rain occurs:",
                [
                    "Ensure proper field drainage to prevent waterlogging",
                    "Check for standing water in fields",
                    "Avoid fertilizer application during heavy rain"
                ]
            ))
            advice.append((
                "🍃",
                "If leaf diseases appear:",
                [
                    "Monitor for leaf blight and rust",
                    "Ensure good air circulation",
                    "Remove affected leaves if necessary"
                ]
            ))
        elif agro_climatic_zone == "Semi-arid":
            advice.append((
                "☀️",
                "If very hot days continue:",
                [
                    "Increase irrigation frequency",
                    "Avoid midday spraying",
                    "Use shade or mulching if possible"
                ]
            ))
            advice.append((
                "💧",
                "If water becomes scarce:",
                [
                    "Prioritize critical growth stages",
                    "Use drip irrigation",
                    "Reduce water loss with mulching"
                ]
            ))
        elif agro_climatic_zone == "Tropical":
            advice.append((
                "🌧️",
                "If heavy rain occurs:",
                [
                    "Ensure proper field drainage to prevent waterlogging",
                    "Check for standing water in fields",
                    "Avoid fertilizer application during heavy rain"
                ]
            ))
            advice.append((
                "🍃",
                "If leaf diseases appear:",
                [
                    "Monitor for leaf blight and rust",
                    "Ensure good air circulation",
                    "Remove affected leaves if necessary"
                ]
            ))
    
    return advice

st.set_page_config(
    page_title="Crop Yield Estimate",
    layout="wide"
)

MODEL_PATH = "models/model.pth"

@st.cache_resource
def load_model():
    """Load the trained PyTorch model and move to appropriate device."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate model architecture first
        model = LSTMYieldModel()
        # Load weights using state_dict (not loading entire model object)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        # Move model to device
        model.to(device)
        # Set model to evaluation mode
        model.eval()
        
        return model, device
    except Exception as e:
        # Re-raise to be caught by outer try-except block
        raise Exception(f"Failed to load model: {str(e)}")

@st.cache_resource
def load_data(device):
    """Load regional agricultural data (internal - not visible to farmers)."""
    try:
        # Internal: Load regional historical data
        # This contains hard-coded regional assumptions:
        # - Regional average climate patterns
        # - Regional soil characteristics  
        # - Historical crop yield averages
        X = np.load("data/X_final.npy")
        y = np.load("data/y_final.npy")
        
        # Internal validation (not shown to farmers)
        assert not np.isnan(X).any(), "X data contains NaN values"
        assert not np.isnan(y).any(), "y data contains NaN values"
        assert X.ndim == 3, f"X must be 3D, got {X.ndim}D"
        assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
        assert len(X) == len(y), f"X and y must have same length, got {len(X)} and {len(y)}"
        
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        
        return X_tensor, y_tensor
    except FileNotFoundError:
        st.error("⚠️ Unable to load agricultural data. Please try again later.")
        st.stop()
    except AssertionError:
        st.error("⚠️ Unable to load agricultural data. Please try again later.")
        st.stop()
    except Exception:
        st.error("⚠️ Unable to load agricultural data. Please try again later.")
        st.stop()

# Safety check for prediction system (internal - not visible to farmers)
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Prediction system is not available. Please contact support.")
    st.stop()

try:
    model, device = load_model()
    X_data, y_data = load_data(device)
except Exception as e:
    st.error("⚠️ Unable to load prediction system. Please try again later.")
    st.stop()

st.header("🌾 Crop Yield Estimate")
st.write("See how much you can expect to harvest and get farming advice based on your crop's growth stage.")

with st.sidebar:
    st.header("📋 Your Farm Details")
    
    # Indian states dropdown (for user familiarity only, not used in ML model)
    indian_states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
    ]
    
    # State to agro-climatic zone mapping (independent of ML model)
    # Maps states to zones to provide location-specific recommendations without requiring exact coordinates
    state_to_zone = {
        # Coastal zones
        "Andhra Pradesh": "Coastal",
        "Goa": "Coastal",
        "Gujarat": "Coastal",
        "Karnataka": "Coastal",
        "Kerala": "Coastal",
        "Maharashtra": "Coastal",
        "Odisha": "Coastal",
        "Tamil Nadu": "Coastal",
        "West Bengal": "Coastal",
        "Puducherry": "Coastal",
        "Lakshadweep": "Coastal",
        "Andaman and Nicobar Islands": "Coastal",
        "Dadra and Nagar Haveli and Daman and Diu": "Coastal",
        
        # Tropical zones
        "Arunachal Pradesh": "Tropical",
        "Assam": "Tropical",
        "Bihar": "Tropical",
        "Chhattisgarh": "Tropical",
        "Jharkhand": "Tropical",
        "Manipur": "Tropical",
        "Meghalaya": "Tropical",
        "Mizoram": "Tropical",
        "Nagaland": "Tropical",
        "Sikkim": "Tropical",
        "Telangana": "Tropical",
        "Tripura": "Tropical",
        
        # Semi-arid zones
        "Haryana": "Semi-arid",
        "Himachal Pradesh": "Semi-arid",
        "Madhya Pradesh": "Semi-arid",
        "Punjab": "Semi-arid",
        "Rajasthan": "Semi-arid",
        "Uttar Pradesh": "Semi-arid",
        "Uttarakhand": "Semi-arid",
        "Chandigarh": "Semi-arid",
        "Delhi": "Semi-arid",
        "Jammu and Kashmir": "Semi-arid",
        "Ladakh": "Semi-arid"
    }
    
    state = st.selectbox(
        "Select your state",
        options=["Select a state"] + indian_states,
        help="Select your state for reference (this does not affect predictions)"
    )
    
    # Get agro-climatic zone for selected state
    # Zones used instead of exact location: provides regional context while maintaining model simplicity
    # IMPORTANT: Only agro_climatic_zone (not state name) can be used as location input to ML model
    # The raw state name must NEVER be passed to the model
    agro_climatic_zone = None
    if state != "Select a state":
        agro_climatic_zone = state_to_zone.get(state, "Tropical")  # Default to Tropical if not found
    
    crop = st.selectbox(
        "What crop are you growing?",
        options=["Rice", "Wheat", "Maize"]
    )
    
    week = st.slider(
        "Current growing season week",
        min_value=1,
        max_value=20,
        value=16,
        help="Select the current week of your growing season (1-20 weeks)"
    )
    
    st.divider()
    st.info("💡 **Tip:** Estimates become more accurate as the season progresses. Early estimates are based on regional averages.")
    
    # Initialize session state for prediction persistence
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    predict_button = st.button("Get Yield Estimate", type="primary")
    
    # Update session state when button is clicked
    if predict_button:
        st.session_state.prediction_made = True

# Internal: Regional assumptions are hard-coded (not visible to farmers)
# - Regional average climate: embedded in historical data
# - Regional soil assumption: embedded in historical data  
# - Historical crop average yield: calculated from y_data (regional average)

if st.session_state.prediction_made:
    try:
        model.eval()
        
        # Location input validation: Only agro_climatic_zone can be used, never raw state name
        # The state variable is for UI only and must never be passed to the model
        if state != "Select a state" and agro_climatic_zone is None:
            st.warning("⚠️ Unable to determine agro-climatic zone. Using default regional data.")
            agro_climatic_zone = "Tropical"  # Safe default
        
        # Display season stage message (no technical details shown)
        if week < 10:
            st.info(f"📅 **Early Season** (Week {week} of 20) - Estimates are based on regional averages and will become more accurate as the season progresses.")
        else:
            st.info(f"📅 **Mid-to-Late Season** (Week {week} of 20) - Estimate based on current crop conditions.")
        
        with torch.no_grad():
            # Slice time series from beginning (first N weeks) for partial-season prediction
            # Convert week to 0-indexed: week 1 -> index 0, week N -> index N-1
            end_week_idx = week  # Selected week (1-indexed)
            
            # Assertions for shape validation
            assert end_week_idx <= X_data.shape[1], f"Week {week} exceeds available data (max: {X_data.shape[1]})"
            assert len(X_data) > 0, "X_data is empty"
            
            # Get sequence from start (week 0) to selected week
            # NOTE: Model input is time series data only. Location (agro_climatic_zone) is not currently
            # used in model architecture, but if needed, only agro_climatic_zone (never state name) should be used.
            X_sample = X_data[0:1, :end_week_idx]  # Shape: (1, seq_len, features)
            
            # Assertions for NaN and shape
            assert not torch.isnan(X_sample).any(), "X_sample contains NaN values"
            assert X_sample.shape[1] > 0, f"Sequence length must be > 0, got {X_sample.shape[1]}"
            
            seq_length = torch.tensor([X_sample.shape[1]], dtype=torch.long)
            
            # Predict yield
            # Model call: Only time series data (X_sample) and sequence length are passed
            # 
            # LOCATION INPUT POLICY:
            # - If location data is ever needed for model input, ONLY agro_climatic_zone should be used
            # - The raw state name (state variable) must NEVER be passed to the model
            # - Current model architecture uses only time series features, not location
            # - agro_climatic_zone is available here if needed for future model enhancements
            predicted_yield = model(X_sample, seq_length)
            raw_pred = predicted_yield.item()
            
            # Assertions for prediction validity
            assert not np.isnan(raw_pred), "Predicted yield is NaN"
            assert np.isfinite(raw_pred), "Predicted yield is not finite"
            
            # Compute average predicted yield across all samples
            # For efficiency, use a subset or all samples
            batch_size = min(100, len(X_data))  # Limit to 100 samples for performance
            X_batch = X_data[:batch_size, :end_week_idx]
            
            # Assertions for batch
            assert not torch.isnan(X_batch).any(), "X_batch contains NaN values"
            assert X_batch.shape[0] > 0, "Batch size must be > 0"
            
            seq_lengths = torch.full((batch_size,), X_batch.shape[1], dtype=torch.long)
            
            # Model call: Only time series data passed, no location input
            # If location is needed: use agro_climatic_zone, never state name
            predictions = model(X_batch, seq_lengths)
            avg_predicted_yield = predictions.mean().item()
            
            # Assertions for average prediction
            assert not np.isnan(avg_predicted_yield), "Average predicted yield is NaN"
            assert np.isfinite(avg_predicted_yield), "Average predicted yield is not finite"
        
        # Internal: Calculate regional historical crop average yield
        # This represents the regional average yield for the selected crop
        # Based on historical data (hard-coded regional assumptions)
        crop_average_kg = y_data.mean().item()
        crop_average_tons = crop_average_kg / 1000
        
        # Assertions for crop average
        assert not np.isnan(crop_average_kg), "Crop average is NaN"
        assert crop_average_kg > 0, f"Crop average must be > 0, got {crop_average_kg}"
        
        # Crop-specific realistic yield ranges (in tons/ha)
        crop_ranges = {
            "Rice": (2.0, 6.0),
            "Wheat": (2.0, 5.5),
            "Maize": (3.0, 8.0)
        }
        min_yield_tons, max_yield_tons = crop_ranges.get(crop, (2.0, 6.0))
        min_yield_kg = min_yield_tons * 1000
        max_yield_kg = max_yield_tons * 1000
        
        # Compute raw yield in kg/ha
        raw_kg = raw_pred * crop_average_kg
        
        # Clamp model-predicted yield to crop-specific realistic bounds
        clamped_model_prediction_kg = np.clip(raw_kg, min_yield_kg, max_yield_kg)
        
        # Apply week-based convergence: early weeks converge toward crop average
        # Use stronger convergence for early weeks to prevent unrealistic values
        total_weeks = 20
        progress = week / total_weeks
        # Use exponential weighting: early weeks heavily weighted toward average
        convergence_weight = progress ** 1.5  # Slower divergence early season
        final_yield_kg = (1 - convergence_weight) * crop_average_kg + convergence_weight * clamped_model_prediction_kg
        
        # Final clamp to ensure realistic range
        final_yield_kg = np.clip(final_yield_kg, min_yield_kg, max_yield_kg)
        
        # Convert to tons for display
        final_tons = final_yield_kg / 1000
        
        # Calculate range for display (not used in calculations)
        margin_pct = 10 + (progress * 5)  # 10% early season, up to 15% later
        confidence_range_low = final_tons * (1 - margin_pct / 100)
        confidence_range_high = final_tons * (1 + margin_pct / 100)
        
        # Get advisory based on converged predicted yield
        advisory = get_yield_advisory(final_yield_kg, crop_average_kg)
        
    except AssertionError as e:
        st.error("⚠️ Unable to generate estimate. Please check your inputs and try again.")
        st.stop()
    except Exception as e:
        st.error("⚠️ Unable to generate estimate. Please try again later.")
        st.stop()
    
    # Calculate trend (comparing current week to previous weeks)
    trend = "Stable"
    total_weeks = 20
    if week > 1:
        # Get previous week prediction for comparison
        with torch.no_grad():
            X_prev = X_data[0:1, :week-1]
            seq_len_prev = torch.tensor([X_prev.shape[1]], dtype=torch.long)
            # Model call: Only time series data passed, no location input
            # If location needed: use agro_climatic_zone, never state name
            pred_prev = model(X_prev, seq_len_prev).item()
            prev_raw_kg = pred_prev * crop_average_kg
            prev_clamped_kg = np.clip(prev_raw_kg, min_yield_kg, max_yield_kg)
            prev_progress = (week - 1) / total_weeks
            prev_convergence_weight = prev_progress ** 1.5
            prev_yield_kg = (1 - prev_convergence_weight) * crop_average_kg + prev_convergence_weight * prev_clamped_kg
            prev_yield_kg = np.clip(prev_yield_kg, min_yield_kg, max_yield_kg)
            prev_tons = prev_yield_kg / 1000
            
            change_pct = ((final_tons - prev_tons) / prev_tons) * 100
            if change_pct > 2:
                trend = "Increasing"
            elif change_pct < -2:
                trend = "Decreasing"
    
    # Calculate yield ratio for health assessment
    yield_ratio = final_tons / crop_average_tons if crop_average_tons > 0 else 1.0
    
    # Calculate yield range for display (display layer only, underlying value unchanged)
    # Range shown instead of single value: accounts for prediction uncertainty and provides realistic expectations
    # Use ±10-15% margin based on week (10% early season, up to 15% later)
    total_weeks = 20
    progress = week / total_weeks
    margin_pct = 10 + (progress * 5)  # 10% early season, up to 15% later
    yield_range_low = final_tons * (1 - margin_pct / 100)
    yield_range_high = final_tons * (1 + margin_pct / 100)
    
    # Determine crop health status
    if yield_ratio >= 0.95:
        health_category = "Growing well"
        health_category_emoji = "🟢"
    elif yield_ratio >= 0.80:
        health_category = "Needs attention"
        health_category_emoji = "🟡"
    else:
        health_category = "Needs help"
        health_category_emoji = "🔴"
    
    # Display results section with all required information - made prominent
    st.divider()
    st.markdown("## 📊 Your Expected Harvest")
    
    # Create a clean layout with key information
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        # Selected State
        if state != "Select a state":
            st.markdown("**📍 Selected State:**")
            st.write(f"{state}")
        else:
            st.markdown("**📍 Selected State:**")
            st.write("*Not selected*")
        
        # Derived Agro-climatic Zone
        st.markdown("**🌍 Agro-climatic Zone:**")
        if agro_climatic_zone:
            st.write(f"{agro_climatic_zone}")
        else:
            st.write("*Not available*")
    
    with info_col2:
        # Yield Range
        st.markdown("**📈 Yield Range:**")
        st.markdown(f"### {yield_range_low:.1f} - {yield_range_high:.1f} t/ha")
        
        # Crop Condition
        st.markdown("**🟢 Crop Condition:**")
        st.markdown(f"### {health_category_emoji} {health_category}")
    
    # This Week's Focus section (moved up, shown for all weeks)
    st.divider()
    st.markdown("### 🗓 What to Do This Week")
    
    # Determine crop health status using yield ratio
    is_healthy = yield_ratio >= 0.95
    is_needs_attention = yield_ratio >= 0.80 and yield_ratio < 0.95
    is_stressed = yield_ratio < 0.80
    
    # Generate weekly focus actions based on week, health status, and crop type
    focus_actions = []
    
    if week < 8:
        # Early season actions
        focus_actions.append("Monitor crop growth daily")
        focus_actions.append("Monitor soil moisture this week")
        if week < 5:
            focus_actions.append("Ensure proper seed germination conditions")
        else:
            focus_actions.append("Watch for early signs of stress")
    elif is_healthy:
        # Healthy crop actions
        focus_actions.append("Continue regular irrigation schedule")
        focus_actions.append("Check leaves for any yellowing or dryness")
        if week >= 10 and week < 16:
            focus_actions.append("Plan fertilizer application if needed")
        focus_actions.append("Monitor crop growth weekly")
    elif is_needs_attention:
        # Needs attention actions
        focus_actions.append("Apply irrigation within 3-5 days")
        focus_actions.append("Monitor soil moisture this week")
        focus_actions.append("Check leaves for yellowing or dryness")
        if crop == "Rice":
            focus_actions.append("Check water level in paddy fields")
        elif crop == "Wheat":
            focus_actions.append("Ensure proper drainage if waterlogged")
        else:  # Maize
            focus_actions.append("Check for signs of water stress")
    else:
        # Stressed crop actions
        focus_actions.append("Apply irrigation within 3-5 days")
        focus_actions.append("Monitor soil moisture this week")
        if crop == "Rice":
            focus_actions.append("Check water level - may need immediate irrigation")
            focus_actions.append("Reduce nitrogen fertilizer if leaves appear dark green")
        elif crop == "Wheat":
            focus_actions.append("Reduce nitrogen fertilizer if leaves appear dark green")
            focus_actions.append("Check for root rot if soil is waterlogged")
        else:  # Maize
            focus_actions.append("Reduce nitrogen fertilizer if leaves appear dark green")
            focus_actions.append("Check for pests or diseases")
    
    # Ensure we have 3-4 actions
    focus_actions = focus_actions[:4]
    
    # Track completed tasks for progress indicator
    completed_count = 0
    
    # Display actions as checkboxes with interactive feedback
    for i, action in enumerate(focus_actions):
        checkbox_key = f"focus_action_{week}_{i}"
        is_checked = st.checkbox(action, key=checkbox_key)
        
        # Track completed tasks
        if is_checked:
            completed_count += 1
            # Show educational feedback when checked
            feedback = get_task_feedback(action)
            st.info(feedback)
    
    # Display progress indicator
    total_tasks = len(focus_actions)
    if completed_count > 0:
        progress_text = f"**{completed_count} of {total_tasks} recommended tasks completed this week**"
        if completed_count == total_tasks:
            st.success(f"✅ {progress_text} - Great job staying on top of your crop care!")
        else:
            st.info(f"📊 {progress_text}")
    
    # Warning line
    st.caption("⚠️ **If ignored:** Yield may reduce in later weeks")
    
    # Zone-based farmer recommendations (rule-based, independent of ML model)
    if agro_climatic_zone:
        st.divider()
        with st.expander("🌍 Zone-Specific Recommendations", expanded=False):
            if agro_climatic_zone == "Semi-arid":
                st.warning("⚠️ **Water Stress Warning:** Your region experiences dry conditions. Water management is critical.")
                st.write("**What to do:**")
                st.write("• Use drip irrigation or mulching")
                st.write("• Irrigate during early morning or evening")
                st.write("• Water deeply but less frequently")
            elif agro_climatic_zone == "Tropical":
                st.info("ℹ️ **Balanced Conditions:** Your region has favorable growing conditions with adequate rainfall.")
                st.write("**What to do:**")
                st.write("• Check crops regularly for pests and diseases")
                st.write("• Maintain proper spacing between plants")
                st.write("• Monitor for fungal diseases during high humidity")
            elif agro_climatic_zone == "Coastal":
                st.warning("⚠️ **High Humidity Warning:** Your coastal region has high humidity levels that can affect crop health.")
                st.write("**What to do:**")
                st.write("• Ensure proper field drainage")
                st.write("• Monitor for fungal and bacterial diseases")
                st.write("• Use raised beds or proper slope for water runoff")
    
    # Typical Weather This Season section (rule-based, descriptive only)
    if agro_climatic_zone:
        st.divider()
        with st.expander("🌦️ Typical Weather This Season", expanded=False):
            weather_description = get_weather_description(crop, agro_climatic_zone)
            st.write(weather_description)
            
            # Possible Weather Risks section (rule-based, emoji indicators)
            weather_risks = get_weather_risks(crop, agro_climatic_zone)
            if weather_risks:
                st.subheader("⚠️ Possible Weather Risks")
                for risk in weather_risks:
                    st.write(risk)
            
            # Conditional Weather Advice section (rule-based, does not affect predictions)
            conditional_advice = get_conditional_weather_advice(crop, agro_climatic_zone)
            if conditional_advice:
                st.subheader("🌱 If Weather Changes, Do This")
                for emoji, condition, actions in conditional_advice:
                    st.write(f"**{emoji} {condition}**")
                    for action in actions:
                        st.write(f"• {action}")
    
    # Crop Condition section (only show from Week 8 onwards) - made prominent
    if week >= 8:
        st.divider()
        st.markdown("## 🟢 Crop Condition")
        
        # Determine health status using yield_ratio with crop-aware messages
        if yield_ratio >= 0.95:
            # Healthy
            health_message = "Your crop is growing well for this stage."
            health_color = "🟢"
            health_display = st.success
        elif yield_ratio >= 0.80:
            # Needs Attention
            health_message = "Your crop is slightly below expected. Monitor closely."
            health_color = "🟡"
            health_display = st.warning
        else:
            # Stressed
            health_message = "Your crop is under stress and needs action."
            health_color = "🔴"
            health_display = st.error
        
        health_display(f"{health_color} **{health_message}**")
        
        # Advisory messages with crop-specific recommendations (limited to 3-4 bullet points)
        st.divider()
        with st.expander("📌 Advisor's Message", expanded=False):
            if yield_ratio >= 0.95:
                # Healthy crop - crop-specific messages
                if crop == "Rice":
                    st.write("Your rice crop is performing well.")
                    st.write("**What to do:**")
                    st.write("• Continue regular irrigation")
                    st.write("• Monitor soil moisture")
                    st.write("• Plan fertilizer if needed")
                elif crop == "Wheat":
                    st.write("Your wheat crop is growing well.")
                    st.write("**What to do:**")
                    st.write("• Continue regular irrigation")
                    st.write("• Monitor soil moisture")
                    st.write("• Check drainage to avoid waterlogging")
                else:  # Maize
                    st.write("Your maize crop is developing nicely.")
                    st.write("**What to do:**")
                    st.write("• Continue regular irrigation")
                    st.write("• Monitor soil moisture")
                    st.write("• Watch for signs of stress")
            elif yield_ratio >= 0.80:
                # Needs Attention - crop-specific messages
                if crop == "Rice":
                    st.write("Your rice crop is growing slower than expected.")
                    st.write("**What to do:**")
                    st.write("• Apply irrigation within 3-5 days")
                    st.write("• Check water level in paddy fields")
                    st.write("• Check leaves for yellowing or dryness")
                elif crop == "Wheat":
                    st.write("Your wheat crop is growing slower than expected.")
                    st.write("**What to do:**")
                    st.write("• Apply irrigation within 3-5 days")
                    st.write("• Ensure proper drainage if waterlogged")
                    st.write("• Check leaves for yellowing or dryness")
                else:  # Maize
                    st.write("Your maize crop is experiencing some stress.")
                    st.write("**What to do:**")
                    st.write("• Apply irrigation within 3-5 days")
                    st.write("• Check for signs of water stress")
                    st.write("• Check leaves for yellowing or dryness")
            else:
                # Stressed crop - crop-specific messages
                if crop == "Rice":
                    st.write("Your rice crop is under stress and needs action.")
                    st.write("**What to do:**")
                    st.write("• Apply irrigation within 3-5 days")
                    st.write("• Check water level in paddy fields immediately")
                    st.write("• Reduce nitrogen fertilizer if leaves appear dark green")
                elif crop == "Wheat":
                    st.write("Your wheat crop is under stress and needs attention.")
                    st.write("**What to do:**")
                    st.write("• Apply irrigation within 3-5 days")
                    st.write("• Reduce nitrogen fertilizer if leaves appear dark green")
                    st.write("• Check for root rot if soil is waterlogged")
                else:  # Maize
                    st.write("Your maize crop is under stress and needs action.")
                    st.write("**What to do:**")
                    st.write("• Apply irrigation within 3-5 days")
                    st.write("• Reduce nitrogen fertilizer if leaves appear dark green")
                    st.write("• Check for pests or diseases")
    
# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "🌾 Crop Yield Estimate | Based on Agricultural Data"
    "</div>",
    unsafe_allow_html=True
)

