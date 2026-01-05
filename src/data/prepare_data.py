"""
Module de préparation des données
Charge, nettoie et prépare les données pour l'entraînement
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataPreparation:
    """Classe pour gérer la préparation des données"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Créer les dossiers si nécessaire
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
    
    def load_raw_data(self):
        """Charge les données brutes (California Housing Dataset)"""
        print("📥 Chargement des données brutes...")
        
        # Charger le dataset
        california = fetch_california_housing(as_frame=True)
        df = california.frame
        
        # Sauvegarder en CSV
        raw_file = os.path.join(self.raw_dir, 'california_housing.csv')
        df.to_csv(raw_file, index=False)
        print(f"✅ Données sauvegardées: {raw_file}")
        
        return df
    
    def clean_data(self, df):
        """Nettoie les données (gestion des valeurs manquantes, outliers)"""
        print("🧹 Nettoyage des données...")
        
        # Afficher les infos de base
        print(f"Shape initiale: {df.shape}")
        print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
        
        # Supprimer les valeurs manquantes (s'il y en a)
        df_clean = df.dropna()
        
        # Supprimer les outliers extrêmes (méthode IQR)
        for col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Shape après nettoyage: {df_clean.shape}")
        
        return df_clean
    
    def create_features(self, df):
        """Crée des features supplémentaires"""
        print("🔧 Création de features...")
        
        df_features = df.copy()
        
        # Feature engineering: ratio chambres/pièces
        df_features['rooms_per_household'] = df_features['AveRooms'] / df_features['AveOccup']
        df_features['bedrooms_per_room'] = df_features['AveBedrms'] / df_features['AveRooms']
        
        # Feature engineering: population density
        df_features['population_per_household'] = df_features['Population'] / df_features['AveOccup']
        
        print(f"Nouvelles features créées: {df_features.shape[1]} colonnes")
        
        return df_features
    
    def split_and_scale_data(self, df, test_size=0.2, random_state=42):
        """Sépare les données en train/test et normalise"""
        print("✂️ Séparation train/test et normalisation...")
        
        # Séparer features et target
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normaliser les features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir en DataFrame pour garder les noms de colonnes
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"Train set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Sauvegarde les données traitées"""
        print("💾 Sauvegarde des données traitées...")
        
        # Sauvegarder les datasets
        X_train.to_csv(os.path.join(self.processed_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.processed_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(self.processed_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(self.processed_dir, 'y_test.csv'), index=False)
        
        # Sauvegarder le scaler
        joblib.dump(self.scaler, os.path.join(self.processed_dir, 'scaler.pkl'))
        
        print("✅ Données traitées sauvegardées!")
    
    def run_pipeline(self):
        """Exécute le pipeline complet de préparation"""
        print("\n" + "="*60)
        print("🚀 DÉMARRAGE DU PIPELINE DE PRÉPARATION DES DONNÉES")
        print("="*60 + "\n")
        
        # 1. Charger les données brutes
        df = self.load_raw_data()
        
        # 2. Nettoyer les données
        df_clean = self.clean_data(df)
        
        # 3. Créer des features
        df_features = self.create_features(df_clean)
        
        # 4. Split et normalisation
        X_train, X_test, y_train, y_test = self.split_and_scale_data(df_features)
        
        # 5. Sauvegarder
        self.save_processed_data(X_train, X_test, y_train, y_test)
        
        print("\n" + "="*60)
        print("✅ PIPELINE DE PRÉPARATION TERMINÉ")
        print("="*60 + "\n")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Exécuter le pipeline de préparation
    prep = DataPreparation()
    X_train, X_test, y_train, y_test = prep.run_pipeline()