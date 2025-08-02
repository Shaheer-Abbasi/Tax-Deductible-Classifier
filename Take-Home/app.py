import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class TaxDeductibleClassifier:
    # Class to classify transactions as tax-deductible or personal based on business rules and machine learning
    # I usesd a Random Forest Classifier with rule-based features and TF-IDF for text descriptions
    # This is because it allows for both structured and unstructured data to be effectively utilized
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.merchant_encoder = LabelEncoder()
        self.business_keywords = [
            'office', 'business', 'work', 'meeting', 'conference', 'client',
            'supplies', 'equipment', 'software', 'training', 'course',
            'coworking', 'professional', 'networking', 'seminar', 'workshop',
            'macbook', 'laptop', 'computer', 'phone', 'internet', 'uber', 'taxi',
            'flight', 'hotel', 'travel', 'mileage', 'gas', 'fuel'
        ]
        
    # This function creates rule-based features for business expense detection
    # It uses keyword matching, merchant-based rules, amount-based heuristics, and description patterns
    # It converts it into numerical features for the model
    def create_business_rules(self, row):
        description = str(row['description']).lower()
        merchant = str(row['merchant']).lower()
        amount = row['amount']
        
        features = {}
        
        # Keyword matching
        business_score = sum(1 for keyword in self.business_keywords if keyword in description or keyword in merchant)
        features['business_keyword_score'] = business_score
        
        # Merchant-based rules
        business_merchants = ['apple', 'staples', 'best buy', 'wework', 'coursera', 'uber']
        personal_merchants = ['target', 'walmart', 'netflix', 'spotify', 'shell']
        
        if any(bm in merchant for bm in business_merchants):
            features['merchant_business_likelihood'] = 1
        elif any(pm in merchant for pm in personal_merchants):
            features['merchant_business_likelihood'] = -1
        else:
            features['merchant_business_likelihood'] = 0
            
        # Amount-based heuristics
        if amount > 500:  # Large purchases more likely business
            features['high_amount'] = 1
        else:
            features['high_amount'] = 0
            
        # Description patterns
        if any(word in description for word in ['for work', 'business', 'office', 'meeting']):
            features['explicit_business'] = 1
        elif any(word in description for word in ['personal', 'home', 'family', 'vacation']):
            features['explicit_personal'] = 1
        else:
            features['explicit_business'] = 0
            
        return features
    
    # This function generates training labels based on business rules and patterns
    # It uses keyword matching, merchant-based rules, and amount-based heuristics
    # It is the core logic for determining if a transaction is tax-deductible
    def generate_training_labels(self, df):
        labels = []
        
        for _, row in df.iterrows():
            description = str(row['description']).lower()
            merchant = str(row['merchant']).lower()
            
            # Strong business indicators
            if any(phrase in description for phrase in [
                'for work', 'business conference', 'office supplies', 
                'macbook purchase for work', 'coworking space', 'online business course'
            ]):
                labels.append(1)  # Deductible
            # Strong personal indicators
            elif any(phrase in description for phrase in [
                'clothing and accessories', 'groceries and household', 
                'streaming subscription', 'books and home items', 'for vacation'
            ]):
                labels.append(0)  # Not deductible
            # Uber rides (could be business)
            elif 'uber' in merchant and 'ride to office' in description:
                labels.append(1)  # Business travel
            # Default heuristics
            elif merchant in ['staples', 'best buy'] and 'supplies' in description:
                labels.append(1)  # Office supplies/equipment
            elif merchant in ['coursera'] or 'course' in description:
                labels.append(1)  # Professional development
            else:
                # Use amount and context as tie-breaker
                if row['amount'] > 200 and any(word in description for word in ['office', 'work', 'business']):
                    labels.append(1)
                else:
                    labels.append(0)
                    
        return np.array(labels)
    

    # This function prepares features for the model
    # Transforms the DataFrame into a format suitable for training i.e. numbers and vectors
    # It includes rule-based features, numerical transformations, merchant encoding, and TF-IDF for text descriptions
    def prepare_features(self, df):
        features_list = []
        
        # Create rule-based features
        for _, row in df.iterrows():
            rule_features = self.create_business_rules(row)
            features_list.append(rule_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add numerical features
        features_df['amount_log'] = np.log1p(df['amount'])
        features_df['amount_normalized'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Add merchant encoding
        merchants_filled = df['merchant'].fillna('unknown')
        try:
            features_df['merchant_encoded'] = self.merchant_encoder.fit_transform(merchants_filled)
        except:
            features_df['merchant_encoded'] = 0
        
        # Add TF-IDF features for descriptions
        descriptions = df['description'].fillna('').astype(str)
        try:
            tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
            tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            features_df = pd.concat([features_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        except:
            pass
        
        return features_df
    
    # This function trains the classifier
    # It generates labels, prepares features, and fits the model
    # The model then recognizes patterns in the data
    # It prints training accuracy and business transaction counts
    def train(self, df):
        # Generate labels
        y = self.generate_training_labels(df)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Train the model
        self.model.fit(X, y)
        
        # Print training summary
        y_pred = self.model.predict(X)
        print(f"Training Accuracy: {accuracy_score(y, y_pred):.3f}")
        print(f"Business transactions identified: {sum(y)} out of {len(y)}")
        
        return X, y
    
    # This function predicts tax-deductible status and generates reasoning
    # It uses the trained model to classify transactions and provides human-readable explanations
    # It returns a list of results with date, merchant, description, amount, deductible status, and reasoning
    # It outputs the results in both CSV and JSON formats
    # The results are also printed in a summary format
    def predict_with_reasoning(self, df):
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        results = []
        for idx, (_, row) in enumerate(df.iterrows()):
            is_deductible = bool(predictions[idx])
            
            # Generate reasoning
            reasoning = self.generate_reasoning(row, is_deductible)
            
            result = {
                'date': str(row['date']),
                'merchant': str(row['merchant']),
                'description': str(row['description']),
                'amount': float(row['amount']),
                'deductible': is_deductible,
                'reason': reasoning
            }
            results.append(result)
        
        return results
    
    def generate_reasoning(self, row, is_deductible):
        """Generate human-readable reasoning for the classification"""
        description = str(row['description']).lower()
        merchant = str(row['merchant']).lower()
        amount = row['amount']
        
        reasons = []
        
        if is_deductible:
            if 'work' in description or 'business' in description:
                reasons.append("explicitly mentioned as work/business related")
            if 'office' in description:
                reasons.append("office-related expense")
            if merchant in ['staples', 'best buy'] and 'supplies' in description:
                reasons.append("office supplies from business vendor")
            if 'coursera' in merchant or 'course' in description:
                reasons.append("professional development/training")
            if 'uber' in merchant and 'office' in description:
                reasons.append("business transportation")
            if amount > 500:
                reasons.append("high-value purchase typical of business expenses")
            
            if not reasons:
                reasons.append("classified as business expense based on pattern analysis")
                
        else:
            if any(word in description for word in ['clothing', 'groceries', 'home', 'personal']):
                reasons.append("personal/household expense")
            if 'netflix' in merchant or 'spotify' in merchant:
                reasons.append("personal entertainment subscription")
            if 'vacation' in description:
                reasons.append("personal travel/vacation")
            if 'target' in merchant or 'walmart' in merchant:
                reasons.append("general retail purchase, likely personal")
            
            if not reasons:
                reasons.append("classified as personal expense based on pattern analysis")
        
        reasoning = f"Tax {'deductible' if is_deductible else 'non-deductible'}: {', '.join(reasons[:2])}"
            
        return reasoning

def main():
    # Load the CSV 
    df = pd.read_csv('sample_transactions-2.csv')
    
    
    classifier = TaxDeductibleClassifier()
    X, y = classifier.train(df)
    
    results = classifier.predict_with_reasoning(df)

    # CSV output
    output_df = pd.DataFrame([
        {
            'date': r['date'],
            'amount': r['amount'],
            'merchant': r['merchant'],
            'description': r['description'],
            'deductible': r['deductible'],
            'reason': r['reason']
        }
        for r in results
    ])
    output_df.to_csv('tax_deductible_predictions.csv', index=False)
    
    # JSON output
    with open('tax_deductible_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    deductible_count = sum(1 for r in results if r['deductible'])
    total_deductible_amount = sum(r['amount'] for r in results if r['deductible'])
    
    print(f"\nSummary:")
    print(f"Total transactions: {len(results)}")
    print(f"Tax-deductible transactions: {deductible_count}")
    print(f"Total deductible amount: ${total_deductible_amount:,.2f}")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for i, result in enumerate(results[:10]):
        status = "✓ DEDUCTIBLE" if result['deductible'] else "✗ PERSONAL"
        print(f"{i+1:2d}. {status} | ${result['amount']:7.2f} | {result['merchant']:12s} | {result['reason']}")

if __name__ == "__main__":
    main()