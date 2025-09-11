# comparison_dashboard.py - ROBUST VERSION
import streamlit as st
import sys
import os
import time
sys.path.append('.')

# Page config
st.set_page_config(
    page_title="DS Assistant [Mistral] Response Comparison",
    layout="wide"
)

# Initialize comparator with error handling
@st.cache_resource
def load_comparator():
    try:
        from compare_responses import ResponseComparator
        return ResponseComparator()
    except Exception as e:
        st.error(f"Failed to load comparator: {e}")
        return None

def main():
    st.title("🔬 DS Notebook Assistant [Mistral] Response Comparison")
    st.markdown("Compare responses between the dual-adapter system and base model")
    
    # Check if comparator loaded successfully
    comparator = load_comparator()
    if comparator is None:
        st.error("Could not initialize the response comparator. Please check your model files and GPU memory.")
        st.info("Make sure your adapters are in `output/adapters` and `output_theory/adapters`")
        return
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., 'Create a function to calculate RMSE' or 'What is regularization?'"
        )
    
    with col2:
        st.markdown("### Settings")
        show_routing = st.checkbox("Show routing decision", value=True)
        show_timing = st.checkbox("Show response times", value=True)
        
        # Add memory info for WSL users
        st.markdown("### System Info")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.info(f"GPU Memory: {gpu_memory:.1f} GB")
            else:
                st.warning("CUDA not available")
        except:
            st.warning("Could not check GPU status")
    
    if st.button("🔬 Compare Responses", type="primary", use_container_width=True):
        if user_input.strip():
            # Add progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with st.spinner("Generating responses..."):
                    status_text.text("Initializing comparison...")
                    progress_bar.progress(10)
                    
                    # Add timeout handling
                    start_time = time.time()
                    results = comparator.compare_responses(user_input.strip())
                    total_time = time.time() - start_time
                    
                    progress_bar.progress(100)
                    status_text.text(f"Comparison completed in {total_time:.1f}s")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Response Comparison")
                    
                    col_adapter, col_base = st.columns(2)
                    
                    with col_adapter:
                        st.markdown("### 🧠 Dual-Adapter System")
                        if show_routing:
                            st.info(f"**Router Decision:** {results['adapter_type'].upper()} adapter selected")
                        if show_timing:
                            st.metric("Response Time", f"{results['adapter_time']:.2f}s")
                        
                        st.markdown("**Response:**")
                        adapter_resp = results['adapter_response']
                        
                        # Check for errors
                        if "Error" in adapter_resp:
                            st.error(adapter_resp)
                        elif results['adapter_type'] == 'code' and '```' in adapter_resp:
                            st.code(adapter_resp.replace('```python', '').replace('```', ''), language='python')
                        else:
                            st.markdown(adapter_resp)
                    
                    with col_base:
                        st.markdown("### 🤖 Base Model")
                        if show_timing:
                            st.metric("Response Time", f"{results['base_time']:.2f}s")
                        
                        st.markdown("**Response:**")
                        base_resp = results['base_response']
                        
                        # Check for errors
                        if "Error" in base_resp:
                            st.error(base_resp)
                        else:
                            st.markdown(base_resp)
                    
                    # Analysis section
                    st.markdown("---")
                    st.subheader("📈 Analysis")
                    
                    col_metrics, col_insights = st.columns([1, 2])
                    
                    with col_metrics:
                        st.metric("Adapter Type", results['adapter_type'].title())
                        length_diff = len(results['adapter_response']) - len(results['base_response'])
                        st.metric("Response Length Diff", f"{length_diff} chars")
                        
                        # Performance metrics
                        if results['adapter_time'] > 0 and results['base_time'] > 0:
                            speedup = results['base_time'] / results['adapter_time']
                            st.metric("Speed Ratio", f"{speedup:.2f}x")
                    
                    with col_insights:
                        st.markdown("**Key Differences:**")
                        insights = comparator.analyze_differences(results)
                        for insight in insights:
                            st.write(f"• {insight}")
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
                st.info("This might be due to GPU memory limits in WSL. Try:")
                st.write("• Restart the Streamlit app")
                st.write("• Use shorter input text")
                st.write("• Check that your adapters are properly saved")
                
                # Clear progress indicators
                try:
                    progress_bar.empty()
                    status_text.empty()
                except:
                    pass
        else:
            st.warning("Please enter a question to compare responses.")
    
    # Add cleanup button for WSL
    if st.button("🧹 Clear GPU Memory", help="Manually clear GPU memory if you encounter issues"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.success("GPU memory cleared")
            else:
                st.info("No CUDA available to clear")
        except Exception as e:
            st.error(f"Could not clear memory: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Optimized for WSL environments • Demonstrates intelligent adapter routing vs base model responses</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()