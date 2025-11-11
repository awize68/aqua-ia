# test_optimizer.py
import sys
import os

# Ajoute le dossier courant au chemin de Python pour être sûr d'importer le bon fichier
sys.path.insert(0, os.getcwd())

print("--- DÉBUT DU TEST DE DIAGNOSTIC ---")

try:
    print("1. Tentative d'import de la classe PlantOptimizer...")
    from optimizer import PlantOptimizer
    print("   -> Importation RÉUSSIE.")

    print("2. Création d'un 'predicteur' factice (dummy) nécessaire pour l'initialisation...")
    from predictive_model import DesalinationPredictor
    dummy_predictor = DesalinationPredictor()
    # On le fait croire qu'il est entraîné pour éviter une erreur
    dummy_predictor.is_trained = True
    print("   -> Predicteur factice créé.")

    print("3. Tentative d'instanciation de l'objet PlantOptimizer...")
    optimizer_instance = PlantOptimizer(dummy_predictor)
    print("   -> Instanciation RÉUSSIE.")

    print("4. Vérification de l'attribut 'chemical_cost_per_mg_l'...")
    # C'est le moment de vérité
    cost_value = optimizer_instance.chemical_cost_per_mg_l
    print(f"   -> SUCCÈS ! L'attribut existe et sa valeur est : {cost_value}")

    print("\n--- TEST RÉUSSI ---")
    print("Le problème ne vient PAS de la classe PlantOptimizer elle-même.")
    print("Le problème vient probablement de la manière dont Streamlit ou dashboard.py l'utilisent.")

except AttributeError as e:
    print(f"\n--- TEST ÉCHOUÉ ---")
    print(f"ERREUR : {e}")
    print("La preuve est faite : l'objet n'a PAS l'attribut. Le problème est DANS le fichier optimizer.py ou son chargement.")
    print("Vérifiez le contenu de votre fichier optimizer.py. Il est possible que votre éditeur ne sauvegarde pas correctement.")

except ImportError as e:
    print(f"\n--- TEST ÉCHOUÉ ---")
    print(f"ERREUR : {e}")
    print("Python ne trouve pas le fichier 'optimizer.py'. Êtes-vous bien dans le bon dossier ?")

except Exception as e:
    print(f"\n--- TEST ÉCHOUÉ ---")
    print(f"ERREUR INATTENDUE : {e}")
