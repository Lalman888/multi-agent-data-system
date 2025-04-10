"""
Utility functions for the Streamlit app
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

def display_quantitative_data_chart(data: Dict[str, Any]):
    """
    Creates visualizations for quantitative data
    
    Args:
        data: Dictionary of quantitative data
    """
    # Convert to dataframe
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    df.index.name = 'Metric'
    df.reset_index(inplace=True)
    
    # Filter only numeric values
    numeric_df = df[pd.to_numeric(df['Value'], errors='coerce').notna()]
    
    if len(numeric_df) > 0:
        # Create bar chart with Plotly
        fig = px.bar(
            numeric_df, 
            x='Metric', 
            y='Value',
            title='Quantitative Metrics',
            labels={'Value': 'Value', 'Metric': 'Metric'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric data available for visualization")

def create_qualitative_data_table(data: Dict[str, str]):
    """
    Creates a formatted table for qualitative data
    
    Args:
        data: Dictionary of qualitative data
    """
    # Convert to dataframe
    df = pd.DataFrame(list(data.items()), columns=['Category', 'Description'])
    
    # Display as table
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        }
    )

def format_analysis_data(analysis: Dict[str, Any]):
    """
    Formats analysis data for better display
    
    Args:
        analysis: Analysis data dictionary
    """
    for section, content in analysis.items():
        st.subheader(section.replace('_', ' ').title())
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'title' in item and 'description' in item:
                    st.markdown(f"**{item['title']}**")
                    st.markdown(item['description'])
                else:
                    st.markdown(f"- {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                st.markdown(f"**{key}**")
                st.markdown(value)
        else:
            st.markdown(content)
        
        st.divider()