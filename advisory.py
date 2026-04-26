def get_yield_advisory(predicted_yield, crop_average, low_threshold=0.85, high_threshold=1.15):
    """
    Generate yield advisory based on predicted yield compared to crop average.
    
    Args:
        predicted_yield: Predicted yield value
        crop_average: Average yield for the crop
        low_threshold: Threshold below average (default 0.85 = 85% of average)
        high_threshold: Threshold above average (default 1.15 = 115% of average)
    
    Returns:
        dict with 'status' and 'message' keys
    """
    ratio = predicted_yield / crop_average
    
    if ratio < low_threshold:
        status = "Low"
        message = "fertilizer"
    elif ratio > high_threshold:
        status = "High"
        message = "ok"
    else:
        status = "Normal"
        message = "care"
    
    return {
        "status": status,
        "message": message,
        "predicted_yield": predicted_yield,
        "crop_average": crop_average,
        "ratio": ratio
    }

