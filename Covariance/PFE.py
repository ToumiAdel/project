import mne
import numpy as np
from moabb.datasets import BNCI2014_001
import matplotlib.pyplot as plt
from pyriemann.utils.base import logm, expm
from scipy.linalg import sqrtm

# Fonction pour transporter un vecteur Delta à travers deux matrices A et B dans l'espace tangent
def transport_vector(Delta, A, B):
    inv_B = np.linalg.inv(B)  # Calcul de l'inverse de B
    A_invB = np.dot(A, inv_B)  # Multiplication matricielle de A et de l'inverse de B
    E = sqrtm(A_invB)  # Racine carrée matricielle de A * inv(B)
    transported_Delta = np.dot(np.dot(E, Delta), E.T)  # Transport de Delta via E
    return transported_Delta  # Retourne Delta transporté

# Fonction pour projeter un vecteur depuis l'espace tangent vers l'espace originel
def inverse_tangent_space_projection(ProjB_A, B):
    sqrt_B = sqrtm(B)  # Racine carrée matricielle de B
    sqrt_inv_B = np.linalg.inv(sqrt_B)  # Inverse de la racine carrée de B
    transformed_ProjB_A = np.dot(np.dot(sqrt_inv_B, ProjB_A), sqrt_inv_B)  # Transformation de ProjB_A
    exp_transformed_ProjB_A = expm(transformed_ProjB_A)  # Exponentielle matricielle de la transformation
    original_A = np.dot(np.dot(sqrt_B, exp_transformed_ProjB_A), sqrt_B)  # Reconstitution de A original
    return original_A  # Retourne A original

# Fonction pour projeter une matrice A dans l'espace tangent défini par B
def tangent_space_projection(A, B):
    sqrt_B = sqrtm(B)  # Racine carrée matricielle de B
    sqrt_inv_B = np.linalg.inv(sqrt_B)  # Inverse de la racine carrée de B
    transformed_A = np.dot(np.dot(sqrt_inv_B, A), sqrt_inv_B)  # Transformation de A
    log_transformed_A = logm(transformed_A)  # Logarithme matriciel de la transformation
    tangent_space_A = np.dot(np.dot(sqrt_B, log_transformed_A), sqrt_B)  # Projeter A dans l'espace tangent
    return tangent_space_A  # Retourne la projection tangentielle de A

# Fonction pour calculer les trajectoires à partir d'un jeu de données EEG
def compute_trajectoires(dataset, subject, session_name, run_name, tmin, tmax, tstride, stride):
    
    # Obtention des données pour un sujet donné
    sessions = dataset.get_data(subjects=[subject])
    raw = sessions[subject][session_name][run_name]
    
    # Extraction des événements et des annotations à partir des données brutes
    events, event_id = mne.events_from_annotations(raw)
    
    # Création des epochs sur la base des événements
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    
    # Initialisation des listes pour stocker les covariances, projections,trqjectoires et labels
    cov_list = []
    projections = []
    label_list = []
    trajectoire = []
    label = []

    # Parcours de chaque epoch
    for i in range(epochs.events.shape[0]):
        epoch = epochs[i]  # Sélection de l'epoch courant
        start = 0
        end = tstride
        cpt = 0
        # Recadrage de l'epoch pour la fenêtre de temps définie
        cropped_epoch = epoch.copy().crop(tmin=0, tmax=tstride)
        # Calcul de la covariance pour le segment recadré
        cov_segment = mne.compute_covariance(cropped_epoch, method='empirical')
        cov_list.append(cov_segment['data'])
        label_list.append(epoch.events[0, -1])
        while end <= tmax - tmin:
            start += stride
            end = start + tstride
            cropped_epoch = epoch.copy().crop(tmin=start, tmax=end)
            if cropped_epoch.get_data().shape[-1] > 1:
                cov_segment = mne.compute_covariance(cropped_epoch, method='empirical')
                cov_list.append(cov_segment['data'])
                label_list.append(epoch.events[0, -1])
                # Calcul de la projection dans l'espace tangent entre les covariances successives
                projections.append(tangent_space_projection(cov_list[i+cpt], cov_list[i+cpt+1]))
                # Transport des vecteurs dans l'espace tangent
                for j in range(i+cpt-1, -1, -1):
                    projections[j] = transport_vector(projections[j], cov_list[i+cpt+1], cov_list[i+cpt])
            cpt += 1
        # Agrégation des projections pour former les trajectoires
        cpt_sum = 0
        for k in range(i, i+cpt):
            cpt_sum += projections[k]
            if (k + 1) % cpt == 0:  # Si le nombre de matrices de covariance atteint cpt
                trajectoire.append(cpt_sum)
                label.append(label_list[i] % cpt) 
                cpt_sum = 0

    # Retourne les listes de covariances, projections, labels, trajectoires et leurs labels associés
    return cov_list, projections, label_list, trajectoire, label

# Exemple d'utilisation de la méthode
dataset = BNCI2014_001()
subject = 1
session_name = "0train"
run_name = "0"
tmin = -1
tmax = 4
tstride = 2
stride = 0.5



cov_list, projections, label_list, trajectoire, label = compute_trajectoires(dataset,subject,session_name,run_name, tmin, tmax, tstride, stride)