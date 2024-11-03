import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from scipy.fft import fft
import pywt
import warnings
warnings.filterwarnings('ignore')

class MarketRhythmModel:
    def __init__(self, ticker, name):
        self.ticker = ticker
        self.name = name
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.ts_cv_scores = None
        self.additional_metrics = None
        self.features = [
            'price_rhythm',
            'volume_rhythm',
            'price_acceleration',
            'volume_trend',
            'price_trend',
            'volatility_regime',
            'trend_strength',
            'direction_consistency',
            'price_volume_harmony',
            'rhythm_breakout'
        ]

    def get_stock_data(self, period="5y", interval="1d"):
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=period, interval=interval)
        if len(df) == 0:
            raise ValueError(f"No data retrieved for {self.ticker}")
        return df

    def calculate_rhythm_features(self, df):
        """Calculate rhythm-based features with simplified calculations"""
        # Basic returns
        df['returns'] = df['Close'].pct_change()
        
        # Basic rhythm features
        df['price_rhythm'] = df['returns'].rolling(window=20, min_periods=5).std()
        df['volume_rhythm'] = df['Volume'].rolling(window=20, min_periods=5).std() / \
                            df['Volume'].rolling(window=20, min_periods=5).mean()
        
        # Trend features
        df['price_acceleration'] = df['returns'].diff().rolling(window=10, min_periods=3).std()
        df['volume_trend'] = df['Volume'].rolling(window=50, min_periods=10).mean() / \
                            df['Volume'].rolling(window=200, min_periods=20).mean()
        df['price_trend'] = df['Close'].rolling(window=50, min_periods=10).mean() / \
                           df['Close'].rolling(window=200, min_periods=20).mean()
        
        # Market state features
        df['volatility_regime'] = df['price_rhythm'].rolling(window=50, min_periods=10).mean()
        df['trend_strength'] = abs(df['price_trend'] - 1)
        df['direction_change'] = (df['returns'] > 0).astype(float)
        df['direction_consistency'] = df['direction_change'].rolling(window=20, min_periods=5).mean()
        
        # Harmony and breakout features
        df['price_volume_harmony'] = df['price_rhythm'] * df['volume_rhythm']
        df['rhythm_breakout'] = (df['price_rhythm'] > 
                               df['price_rhythm'].rolling(window=50, min_periods=10).mean()).astype(float)
        
        return df.dropna()

    def create_target_variable(self, df, forward_window=5):
        """Create target variable for prediction"""
        df['future_return'] = df['Close'].shift(-forward_window).pct_change(forward_window)
        df['target'] = (df['future_return'] > 0).astype(float)
        return df.dropna()

    def perform_time_series_cv(self, X, y):
        """Perform time-series specific cross-validation with additional metrics"""
        n_splits = min(5, len(X) // 100)
        tscv = TimeSeriesSplit(n_splits=max(2, n_splits))
        scores = []
        additional_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < 50:
                continue
                
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Original accuracy score
            score = model.score(X_test_scaled, y_test)
            scores.append(score)
            
            # Additional metrics
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            additional_metrics['precision'].append(precision_score(y_test, y_pred))
            additional_metrics['recall'].append(recall_score(y_test, y_pred))
            additional_metrics['f1'].append(f1_score(y_test, y_pred))
            additional_metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
        
        self.additional_metrics = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in additional_metrics.items()
        }
        
        return scores

    def train_model(self, df):
        """Train the rhythm analysis model"""
        X = df[self.features].copy()
        y = df['target'].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Perform time-series cross-validation
        self.ts_cv_scores = self.perform_time_series_cv(X, y)
        
        # Train final model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_scaled, y

    def predict_with_confidence(self, data):
        """Make predictions with confidence scores"""
        data = data[self.features].copy()
        data = data.fillna(method='ffill').fillna(method='bfill')
        data_scaled = self.scaler.transform(data)
        probabilities = self.model.predict_proba(data_scaled)
        predictions = self.model.predict(data_scaled)
        confidence = np.max(probabilities, axis=1)
        return predictions, probabilities, confidence

def plot_market_rhythm(df, title):
    """Create visualization of market rhythm patterns"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'{title} Market Rhythm Analysis', fontsize=16)
    
    # Price and Volume Rhythm
    ax1.plot(df.index, df['price_rhythm'], label='Price Rhythm', color='blue')
    ax1.plot(df.index, df['volume_rhythm'], label='Volume Rhythm', color='red')
    ax1.set_title('Price and Volume Rhythm Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Price-Volume Harmony
    ax2.plot(df.index, df['price_volume_harmony'], label='Price-Volume Harmony', color='green')
    ax2.set_title('Price-Volume Harmony Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_stock(ticker, name):
    """Perform complete analysis for a single stock"""
    try:
        print(f"Initializing analysis for {name}...")
        model = MarketRhythmModel(ticker, name)
        
        print("Retrieving data...")
        df = model.get_stock_data()
        
        print("Calculating features...")
        df = model.calculate_rhythm_features(df)
        
        print("Creating target variables...")
        df = model.create_target_variable(df)
        
        print("Training model...")
        X_scaled, y = model.train_model(df)
        
        latest_data = df.iloc[-1:].copy()
        pred, prob, conf = model.predict_with_confidence(latest_data)
        
        print("Creating visualizations...")
        plot_market_rhythm(df, name)
        
        return {
            'name': name,
            'ts_cv_scores_mean': np.mean(model.ts_cv_scores) if model.ts_cv_scores else None,
            'ts_cv_scores_std': np.std(model.ts_cv_scores) if model.ts_cv_scores else None,
            'latest_prediction': 'Increase' if pred[0] == 1 else 'Decrease',
            'confidence': conf[0],
            'up_probability': prob[0][1],
            'feature_importance': model.feature_importance,
            'additional_metrics': model.additional_metrics
        }
        
    except Exception as e:
        print(f"Error analyzing {name}: {str(e)}")
        return None

def analyze_magnificent_seven():
    """Analyze all Magnificent Seven stocks"""
    stocks = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet',
        'AMZN': 'Amazon',
        'NVDA': 'NVIDIA',
        'META': 'Meta',
        'TSLA': 'Tesla'
    }
    
    results = {}
    
    for ticker, name in stocks.items():
        print(f"\nAnalyzing {name} ({ticker})...")
        result = analyze_stock(ticker, name)
        if result:
            results[ticker] = result
            
            print(f"\n{result['name']} ({ticker}):")
            if result['ts_cv_scores_mean'] is not None:
                print(f"Time-Series CV Score: {result['ts_cv_scores_mean']:.3f} "
                      f"(±{result['ts_cv_scores_std']:.3f})")
            print(f"Latest Prediction: {result['latest_prediction']}")
            print(f"Prediction Confidence: {result['confidence']:.2f}")
            print(f"Probability of Increase: {result['up_probability']:.2f}")
            
            # Print additional metrics
            if result['additional_metrics']:
                print("\nAdditional Performance Metrics:")
                for metric, values in result['additional_metrics'].items():
                    print(f"{metric.upper()}: {values['mean']:.3f} (±{values['std']:.3f})")
            
            print("\nTop 5 Important Features:")
            print(result['feature_importance'].head())
            print("-" * 50)
    
    return results

if __name__ == "__main__":
    results = analyze_magnificent_seven()
