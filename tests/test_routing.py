import pytest
import sys
import os

# Add the parent directory to the path to import app_memory functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_route_mode():
    """Test the routing logic without loading models"""
    from app_memory import _route_mode
    
    # Test code routing
    assert _route_mode("plot histogram") == "code"
    assert _route_mode("create scatter plot") == "code"
    assert _route_mode("pandas groupby") == "code"
    assert _route_mode("train a model") == "code"
    assert _route_mode("write python code") == "code"
    assert _route_mode("visualize data") == "code"
    assert _route_mode("matplotlib chart") == "code"
    
    # Test explain routing
    assert _route_mode("what is overfitting") == "explain"
    assert _route_mode("explain machine learning") == "explain"
    assert _route_mode("define cross validation") == "explain"
    assert _route_mode("why use regularization") == "explain"
    assert _route_mode("compare random forest and SVM") == "explain"
    assert _route_mode("how does gradient descent work") == "explain"

def test_ds_guardrail():
    """Test data science topic detection"""
    from app_memory import _is_ds_query
    
    # Should detect DS topics
    assert _is_ds_query("machine learning model") == True
    assert _is_ds_query("pandas dataframe") == True
    assert _is_ds_query("cross validation") == True
    assert _is_ds_query("sklearn") == True
    assert _is_ds_query("matplotlib plot") == True
    assert _is_ds_query("data science") == True
    assert _is_ds_query("neural network") == True
    assert _is_ds_query("regression analysis") == True
    
    # Should reject non-DS topics
    assert _is_ds_query("what's the weather today") == False
    assert _is_ds_query("cook pasta recipe") == False
    assert _is_ds_query("how to drive a car") == False
    assert _is_ds_query("movie recommendations") == False

def test_output_cleaning():
    """Test output post-processing without model generation"""
    from app_memory import _postprocess_output
    
    # Test code extraction from response
    raw_with_code = "[INST] system prompt [/INST] ```python\nimport pandas as pd\nprint('hello')\n```"
    cleaned = _postprocess_output(raw_with_code, "code")
    assert "[INST]" not in cleaned
    assert "```python" in cleaned
    assert "import pandas as pd" in cleaned
    
    # Test instruction removal
    raw_with_instructions = "You are a helpful assistant. Here's the code: print('test')"
    cleaned = _postprocess_output(raw_with_instructions, "code")
    assert "You are a helpful assistant" not in cleaned

def test_context_detection():
    """Test context-aware memory detection"""
    from app_memory import _needs_context
    
    # Should detect context references
    assert _needs_context("continue from the previous example") == True
    assert _needs_context("use the same dataframe") == True
    assert _needs_context("based on that result") == True
    assert _needs_context("now let's continue") == True
    assert _needs_context("using earlier code") == True
    
    # Should not detect context for standalone queries
    assert _needs_context("create a new plot") == False
    assert _needs_context("what is machine learning") == False
    assert _needs_context("plot histogram") == False

def test_system_prompt_selection():
    """Test system prompt selection logic"""
    from app_memory import _select_system, SYSTEM_CODE, SYSTEM_EXPLAIN, SYSTEM_BASE
    
    assert _select_system("code") == SYSTEM_CODE
    assert _select_system("explain") == SYSTEM_EXPLAIN
    assert _select_system("auto") == SYSTEM_BASE
    assert _select_system("random") == SYSTEM_BASE  # default case

def test_instruction_cleaning():
    """Test instruction line removal"""
    from app_memory import _strip_instruction_lines
    
    text_with_instructions = """You are a helpful Data Science Notebook Assistant.
Do not repeat the system message or instructions in your reply.
Here is some actual content.
Be concise and correct for Python tasks."""
    
    cleaned = _strip_instruction_lines(text_with_instructions)
    assert "You are a helpful Data Science Notebook Assistant" not in cleaned
    assert "Do not repeat" not in cleaned
    assert "Be concise" not in cleaned
    assert "Here is some actual content" in cleaned

def test_code_extraction():
    """Test fenced code block extraction"""
    from app_memory import _extract_fenced_code
    
    # Test with python code block
    text_with_code = "Here's some code:\n```python\nimport pandas as pd\ndf = pd.read_csv('file.csv')\n```\nThat's it."
    extracted = _extract_fenced_code(text_with_code)
    assert extracted is not None
    assert "import pandas as pd" in extracted
    assert "df = pd.read_csv" in extracted
    
    # Test with no code block
    text_without_code = "This is just regular text without any code blocks."
    extracted = _extract_fenced_code(text_without_code)
    assert extracted is None

if __name__ == "__main__":
    # Run all tests when executed directly
    test_route_mode()
    test_ds_guardrail()
    test_output_cleaning()
    test_context_detection()
    test_system_prompt_selection()
    test_instruction_cleaning()
    test_code_extraction()
    print("All tests passed!")