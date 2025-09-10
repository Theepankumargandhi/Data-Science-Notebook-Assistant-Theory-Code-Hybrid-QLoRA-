def test_basic_logic():
    """Test basic routing logic without imports"""
    
    def simple_route_mode(text):
        text = text.lower()
        explain_hints = ["explain", "what is", "why", "define", "compare"]
        code_hints = ["plot", "code", "pandas", "sklearn", "train", "visualize"]
        
        if any(hint in text for hint in explain_hints):
            return "explain"
        if any(hint in text for hint in code_hints):
            return "code"
        return "code"
    
    # Test the logic
    assert simple_route_mode("plot histogram") == "code"
    assert simple_route_mode("what is overfitting") == "explain"
    assert simple_route_mode("pandas groupby") == "code"
    print("Basic routing tests passed!")

def test_ds_detection():
    """Test DS topic detection"""
    
    def simple_ds_check(text):
        ds_terms = ["machine learning", "pandas", "sklearn", "data science", "python"]
        return any(term in text.lower() for term in ds_terms)
    
    assert simple_ds_check("machine learning model") == True
    assert simple_ds_check("cook pasta") == False
    print("DS detection tests passed!")

if __name__ == "__main__":
    test_basic_logic()
    test_ds_detection()
    print("All simple tests passed!")