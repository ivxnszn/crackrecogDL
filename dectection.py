import cv2
import numpy as np

# Charger l'image depuis un fichier
image_path = '/Users/ivanbelgacem/Desktop/Coding/imageML/IMAGES2'  # Remplacez par le chemin de votre image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Appliquer la détection de contours avec l'algorithme de Canny
edges = cv2.Canny(blurred, 30, 150)

# Dilater les contours pour les rendre plus visibles
dilated_edges = cv2.dilate(edges, None, iterations=2)

# Afficher l'image originale et les contours détectés
cv2.imshow('Image Originale', image)
cv2.imshow('Contours de fissures', dilated_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
