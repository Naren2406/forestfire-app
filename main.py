# Import necessary libraries
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import meteostat
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from collections import Counter
import folium
from streamlit_folium import folium_static
from sklearn.impute import SimpleImputer


# Function to fetch data from Google Custom Search API
def search_google(query, api_key, cse_id):
    search_url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    return response.json()

# Function to fetch geocoding data from Positionstack API
def fetch_positionstack_data(api_key, location):
    url = f"http://api.positionstack.com/v1/forward?access_key={api_key}&query={location}"
    response = requests.get(url)
    
    st.write("Positionstack API Response Status Code:", response.status_code)
    st.write("Positionstack API Response Text:", response.text)
    
    try:
        data = response.json()
    except ValueError:
        st.error("Error: Unable to decode the JSON response from Positionstack.")
        st.write("Response text:", response.text)  # Print the response text for debugging
        return None
    
    if 'data' not in data or len(data['data']) == 0:
        st.error("Error: Received empty response from Positionstack.")
        return None
    
    return data['data']


# Function to fetch weather data from OpenWeather API
def fetch_weather_data(api_key, lat, lon, positionstack_api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if 'main' in data:
        # Get detailed location information
        positionstack_data = fetch_positionstack_data(positionstack_api_key, f"{lat},{lon}")
        if positionstack_data is None:
            state = country = locality = region = county = 0
        else:
            state = positionstack_data[0].get('region', 0)
            country = positionstack_data[0].get('country', 0)
            locality = positionstack_data[0].get('locality', 0)
            region = positionstack_data[0].get('region', 0)
            county = positionstack_data[0].get('county', 0)

        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon'],
            'precipitation': np.random.uniform(0, 10),  # Adding default synthetic precipitation data
            'past_fires': np.random.randint(0, 2),      # Adding default synthetic past fires data
            'state': state if isinstance(state, (int, float)) else 0,
            'country': country if isinstance(country, (int, float)) else 0,
            'locality': locality if isinstance(locality, (int, float)) else 0,
            'region': region if isinstance(region, (int, float)) else 0,
            'county': county if isinstance(county, (int, float)) else 0,
        }
    else:
        st.error("Error: Unable to fetch data. Please check the API key and location.")
        st.write("Response text:", response.text)  # Print the response text for debugging
        return None


# Function to check for missing or constant values in the dataset
def check_data(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    st.write("Missing values in each column:\n", missing_values)
    
    # Check for constant values
    unique_values = df.nunique()
    st.write("Number of unique values in each column:\n", unique_values)

    constant_columns = unique_values[unique_values == 1].index
    st.write("Columns with constant values:\n", constant_columns)
    
#PREPROSSESS DATA
def preprocess_data(df):
    df = df.dropna(subset=['temperature', 'humidity', 'wind_speed', 'precipitation', 'past_fires'])
    df['temperature'] = df['temperature'].astype(float)
    df['humidity'] = df['humidity'].astype(float)
    df['wind_speed'] = df['wind_speed'].astype(float)
    df['precipitation'] = df['precipitation'].astype(float)
    df['past_fires'] = df['past_fires'].astype(int)
    df['state'] = df['state'].astype(str)
    df['country'] = df['country'].astype(str)
    df['locality'] = df['locality'].astype(str)
    df['region'] = df['region'].astype(str)
    df['county'] = df['county'].astype(str)
    df['fire_risk'] = df['fire_risk'].astype(int)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df[['temperature', 'humidity', 'wind_speed', 'precipitation', 'past_fires']] = imputer.fit_transform(
        df[['temperature', 'humidity', 'wind_speed', 'precipitation', 'past_fires']])
    return df

# Add geographic data
def add_geographic_data(df):
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    return geo_df

# Generate Synthetic Data
def generate_synthetic_data(num_samples):
    np.random.seed(42)
    synthetic_data = pd.DataFrame({
        'temperature': np.random.uniform(15, 35, num_samples),
        'humidity': np.random.uniform(30, 90, num_samples),
        'wind_speed': np.random.uniform(0, 15, num_samples),
        'precipitation': np.random.uniform(0, 20, num_samples),
        'past_fires': np.random.randint(0, 2, num_samples),
        'fire_risk': np.random.randint(0, 3, num_samples)  # Assuming 3 classes for fire risk
    })
    return synthetic_data

# Integrate Real and Synthetic Data
def integrate_data(real_data, synthetic_data):
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
    return combined_data

# Feature Engineering
def feature_engineering(df):
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['wind_precipitation'] = df['wind_speed'] * df['precipitation']
    df['temp_wind'] = df['temperature'] * df['wind_speed']
    df['humidity_precipitation'] = df['humidity'] * df['precipitation']
    return df

# Handling Imbalanced Data for Multi-Class using SMOTE
def balance_data(X, y):
    counter = Counter(y)
    max_count = max(counter.values())
    sampling_strategy = {class_label: max_count for class_label in counter}
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Hyperparameter Tuning for XGBoost
def xgb_tuning(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    xgb_model = xgb.XGBClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

#Calculating Fire Risk
def calculate_fire_risk(model, weather_data, feature_names):
    # Create a DataFrame from the weather data
    X_new = pd.DataFrame([{
        'temperature': weather_data['temperature'],
        'humidity': weather_data['humidity'],
        'wind_speed': weather_data['wind_speed'],
        'precipitation': weather_data['precipitation'],
        'past_fires': weather_data['past_fires'],
        'latitude': weather_data.get('latitude', 0),  # Default value if not present
        'longitude': weather_data.get('longitude', 0),  # Default value if not present
        'state': weather_data.get('state', 'Unknown'),  # Default value if not present
        'country': weather_data.get('country', 'Unknown'),  # Default value if not present
        'locality': weather_data.get('locality', 'Unknown'),  # Default value if not present
        'region': weather_data.get('region', 'Unknown'),  # Default value if not present
        'county': weather_data.get('county', 'Unknown')  # Default value if not present
    }])
    
    # Apply the same feature engineering steps to the new data
    X_new['temp_humidity'] = X_new['temperature'] * X_new['humidity']
    X_new['wind_precipitation'] = X_new['wind_speed'] * X_new['precipitation']
    X_new['temp_wind'] = X_new['temperature'] * X_new['wind_speed']
    X_new['humidity_precipitation'] = X_new['humidity'] * X_new['precipitation']
    
    # Ensure the features match the training data
    missing_features = [feature for feature in feature_names if feature not in X_new.columns]
    if missing_features:
        raise KeyError(f"Missing features in prediction data: {missing_features}")
    
    X_new = X_new[feature_names]
    
    # Print feature names for debugging
    print("Prediction Features:", X_new.columns)
    
    # Make predictions using the trained model
    fire_risk = model.predict(X_new)
    return fire_risk[0]

# Training Model with XGBoost
def train_xgb_model_with_tuning(df):
    df = feature_engineering(df)
    X = df.drop(columns=['fire_risk'])
    y = df['fire_risk']

    # Verify the features after feature engineering
    feature_names = list(X.columns)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train, y_train = balance_data(X_train, y_train)
    
    # Convert X_train and X_test back to DataFrame to maintain column names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # Print feature names for debugging
    print("Training Features:", X_train.columns)
    
    model = xgb_tuning(X_train, y_train)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, feature_names

# Function to fetch previous fires data
def fetch_previous_fires_data(location, api_key, cse_id):
    if not api_key or not cse_id:
        st.error("Google API key and Custom Search Engine ID must be provided.")
        return []
    
    query = f"previous fires near {location}"
    try:
        search_results = search_google(query, api_key, cse_id)
        if search_results and 'items' in search_results:
            fire_incidents = []
            for result in search_results['items']:
                fire_incidents.append({
                    'title': result['title'],
                    'link': result['link'],
                    'snippet': result['snippet']
                })
            return fire_incidents
        else:
            st.error("Error: No search results found for previous fire data.")
            return []
    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")
        return []
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return []

# Visualize Previous Fires Data
def visualize_previous_fires_data(fire_incidents):
    if not fire_incidents:
        st.write("No previous fire incidents data available.")
        return

    for incident in fire_incidents:
        # Confirm the structure of the incident dictionary
        st.write("Incident Data:", incident)
        
        title = incident.get('title', 'Unknown Title')
        link = incident.get('link', '#')
        snippet = incident.get('snippet', 'No description available.')

        st.write(f"**{title}**")
        st.write(f"[Read more]({link})")
        st.write(f"*{snippet}*")
        st.write("---")
        
# Function to visualize data
def visualize_data(df):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Temperature Distribution
    sns.histplot(df['temperature'], kde=True, ax=axes[0, 0], color='orange')
    axes[0, 0].set_title('Temperature Distribution')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Humidity Distribution
    sns.histplot(df['humidity'], kde=True, ax=axes[0, 1], color='blue')
    axes[0, 1].set_title('Humidity Distribution')
    axes[0, 1].set_xlabel('Humidity (%)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Wind Speed Distribution
    sns.histplot(df['wind_speed'], kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Wind Speed Distribution')
    axes[1, 0].set_xlabel('Wind Speed (m/s)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Precipitation Distribution
    sns.histplot(df['precipitation'], kde=True, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Precipitation Distribution')
    axes[1, 1].set_xlabel('Precipitation (mm)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)

# Visualize Geographic Data
def visualize_geographic_data(geo_combined_df, lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        raise ValueError("Latitude and Longitude values cannot be NaN.")
    
    # Create a map centered around the location
    m = folium.Map(location=[lat, lon], zoom_start=13)

    # Add a marker for the location
    folium.CircleMarker(
        location=(lat, lon),
        radius=10,
        popup=f"Lat: {lat}, Lon: {lon}",
        color="blue",
        fill=True,
        fill_color="blue"
    ).add_to(m)
    
    folium_static(m)

#TO GENERATE CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

#TO PLOT GRAPH
def plot_feature_importance(model, feature_names):
    importance = model.get_booster().get_score(importance_type="weight")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [k for k, v in sorted_importance]
    scores = [v for k, v in sorted_importance]
    
    plt.figure(figsize=(10, 7))
    plt.barh(features, scores, color='skyblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    st.pyplot(plt)



# Main function
def main():
    st.title("🌲🔥 Wildfire Risk Calculator 🔥🌲")
    st.markdown("Stay ahead of the fire with accurate predictions. Enter the details below to assess the risk.")

    api_key_openweather = st.text_input("🔑 Enter OpenWeatherMap API Key:")
    positionstack_api_key = st.text_input("🔑 Enter Positionstack API Key:")
    google_api_key = st.text_input("🔑 Enter Google API Key:")
    cse_id = st.text_input("🔑 Enter Google Custom Search Engine ID:")
    location = st.text_input("📍 Enter Location:")

    if st.button("Find Locations"):
        if len(location) < 3:
            st.error("Location input must be at least 3 characters long. Please enter a valid location.")
        else:
            with st.spinner("Fetching data... Hang tight! 🤖"):
                positionstack_data = fetch_positionstack_data(positionstack_api_key, location)
                if positionstack_data:
                    st.session_state.positionstack_data = positionstack_data
                    st.session_state.locations = [f"{loc['name']}, {loc['region']}, {loc['country']}" for loc in positionstack_data]
                    st.session_state.selected_location = st.session_state.locations[0]

    if "locations" in st.session_state:
        selected_location = st.selectbox("Select the correct location:", st.session_state.locations, index=st.session_state.locations.index(st.session_state.selected_location))
        st.session_state.selected_location = selected_location

    if st.button("Calculate Fire Risk"):
        if "positionstack_data" in st.session_state:
            selected_data = next(loc for loc in st.session_state.positionstack_data if f"{loc['name']}, {loc['region']}, {loc['country']}" == st.session_state.selected_location)
            lat = selected_data['latitude']
            lon = selected_data['longitude']

            with st.spinner("Fetching weather data..."):
                weather_data = fetch_weather_data(api_key_openweather, lat, lon, positionstack_api_key)

            if weather_data:
                st.write("Weather data fetched successfully.")
                real_data = pd.DataFrame([weather_data])
                real_data['fire_risk'] = 0  # Placeholder for fire risk
                real_data = preprocess_data(real_data)
                geo_df = add_geographic_data(real_data)
                
                st.info("Generating synthetic data... 🔄")
                synthetic_df = generate_synthetic_data(1000)
                combined_data = integrate_data(real_data, synthetic_df)
                geo_combined_df = add_geographic_data(combined_data)

                geo_combined_df.to_file('preprocessed_data.geojson', driver='GeoJSON')
                st.success("Data preprocessing complete! 🛠️")

                st.info("Fetching previous fire incidents... 🔍")
                fire_incidents = fetch_previous_fires_data(location, google_api_key, cse_id)
                
                st.info("Training the model... This might take a few seconds. ⏳")
                model, accuracy, X_test, y_test, feature_names = train_xgb_model_with_tuning(combined_data)

                if model:
                    fire_risk = calculate_fire_risk(model, weather_data, feature_names)
                    risk_level = ["Low", "Moderate", "High"][fire_risk]
                    st.write(f"🌍 **Location**: {st.session_state.selected_location}")
                    st.write(f"🔥 **Fire Risk Level**: {risk_level}")
                    st.success("Model training complete! 🎉")

                    # Debugging printouts to identify feature names mismatch
                    st.write("## Debugging Printouts")
                    st.write("### Training Features")
                    st.write(feature_names)
                    st.write("### Prediction Features")
                    st.write(list(weather_data.keys()))

                    # Plot Confusion Matrix
                    y_pred = model.predict(X_test)
                    st.write("## Confusion Matrix")
                    plot_confusion_matrix(y_test, y_pred)
                    
                    # Plot Feature Importance
                    st.write("## Feature Importance")
                    plot_feature_importance(model, feature_names)
                    
                else:
                    st.warning("Not enough data to train the model. Please add more data points.")
                
                st.info("Generating visualizations... 📊")
                visualize_data(combined_data)

                # Display the map for the selected location
                visualize_geographic_data(geo_combined_df, lat, lon)

                # Display previous fire incidents
                visualize_previous_fires_data(fire_incidents)

if __name__ == "__main__":
    main()
