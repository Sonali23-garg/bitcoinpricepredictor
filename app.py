import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import streamlit as st

# Page configuration
st.set_page_config(page_title="BTC Predictor", layout="wide")
st.title("🚀 Bitcoin Price Predictor")
st.markdown("---")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["📊 Data Analysis", "🤖 Model Training", "🔮 Predictions"])

# Create output directory
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# ==================== PAGE 1: DATA ANALYSIS ====================
if page == "📊 Data Analysis":
    st.header("Data Analysis & Exploration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your BTC CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Store in session state
        st.session_state.df = df
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Dataset Info")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Data Types")
        st.dataframe(df.dtypes, use_container_width=True)
        
        # Visualizations
        st.subheader("Price Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(df["Open"], df["Close"], alpha=0.6, color='blue')
            ax.set_xlabel("Open Price")
            ax.set_ylabel("Close Price")
            ax.set_title("Open vs Close Price")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(df["High"], df["Low"], alpha=0.6, color='green')
            ax.set_xlabel("High Price")
            ax.set_ylabel("Low Price")
            ax.set_title("High vs Low Price")
            st.pyplot(fig)
        
        # Additional analysis
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            st.subheader("Time Series Analysis")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Datetime'], df['Close'], linewidth=2, color='darkblue')
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price (USD)")
            ax.set_title("Bitcoin Price Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload a CSV file to begin analysis")

# ==================== PAGE 2: MODEL TRAINING ====================
elif page == "🤖 Model Training":
    st.header("Train Machine Learning Model")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data in the Data Analysis section first")
    else:
        df = st.session_state.df.copy()
        
        st.subheader("Feature Engineering")
        
        # Prepare data
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.sort_values('Datetime').reset_index(drop=True)
        
        df['returns'] = df['Close'].pct_change()
        df['target'] = np.where(df['returns'].shift(-1) > 0, 1, 0)
        df = df.dropna()
        
        # Extract temporal features
        if 'Datetime' in df.columns:
            df['hour'] = df['Datetime'].dt.hour
            df['day'] = df['Datetime'].dt.dayofweek
        
        df['rsi'] = 50  # Placeholder for RSI
        
        # Feature selection
        st.write("Selected Features:")
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour', 'day']
        st.write(features)
        
        X = df[features]
        y = df['target']
        
        # Model hyperparameters
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20)
        with col2:
            n_estimators = st.slider("Number of Trees", 50, 300, 100)
        with col3:
            random_state = st.number_input("Random State", value=42)
        
        # Train model button
        if st.button("🎯 Train Model", key="train"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                
                # Save model
                joblib.dump(model, output_dir / 'btc_predictor_model.pkl')
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                st.success("✅ Model trained and saved!")
        
        # Display results if model exists
        if 'model' in st.session_state:
            st.subheader("Model Performance")
            
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Test Samples", len(y_test))
            
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

# ==================== PAGE 3: PREDICTIONS ====================
elif page == "🔮 Predictions":
    st.header("Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model in the Model Training section first")
    else:
        st.subheader("Input Features for Prediction")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            open_price = st.number_input("Open Price", value=45000.0)
            high_price = st.number_input("High Price", value=46000.0)
            low_price = st.number_input("Low Price", value=44000.0)
        
        with col2:
            close_price = st.number_input("Close Price", value=45500.0)
            volume = st.number_input("Volume", value=1000000.0)
            hour = st.slider("Hour (0-23)", 0, 23, 12)
        
        with col3:
            day = st.slider("Day of Week (0-6)", 0, 6, 0)
            st.write("0=Monday, 6=Sunday")
        
        if st.button("📈 Predict Price Movement", key="predict"):
            model = st.session_state.model
            
            # Create prediction input
            input_data = np.array([[open_price, high_price, low_price, close_price, volume, hour, day]])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.success("📈 Price will likely GO UP", icon="✅")
                else:
                    st.error("📉 Price will likely GO DOWN", icon="❌")
            
            with col2:
                st.metric("Confidence", f"{max(probability)*100:.2f}%")
            
            # Probability breakdown
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Outcome': ['Price Up', 'Price Down'],
                'Probability': [probability[1]*100, probability[0]*100]
            })
            st.bar_chart(prob_df.set_index('Outcome'))

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | BTC Predictor v1.0")
