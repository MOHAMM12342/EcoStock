# Tickets seuls et composition des produits — 18 mois

- **944 451 tickets**, du 1er janvier 2025 au 30 juin 2026, sur 546 jours.
- `tickets.csv` : une ligne par ticket, date à la seconde avec fuseau, canal, commande lisible, produits et quantités en JSON. Aucun agrégat temporel, cible future, viande calculée, remplissage par zéro ou événement oracle dans ce fichier.
- `composition_produits.csv` : nombre de pièces de viande par unité commerciale, nom du type Hamb/Whopper/Master/Poulet, alternative végétale et indication de menu avec frites/boisson.
- `extrait_40_tickets.csv` : petit aperçu non représentatif.
- `verification.json` : contrôles après relecture du fichier complet.

La jointure se fait par produit_id : multiplier quantite dans produits_json par nombre_pieces_viande dans la composition et additionner par type_viande. Une valeur 3 pour M03 signifie trois menus Whopper, donc trois pièces Whopper dans cette nomenclature fictive.

Les tickets ont tous un contenu non vide et toutes les quantités commandées sont strictement positives. Une boisson ou des frites contiennent zéro viande ; un burger végétal contient une galette végétale et zéro viande ; un burger poulet contient une pièce Poulet et zéro bœuf. Ces cas ne sont pas des commandes vides. Les zéros éventuellement présents dans la table de composition décrivent une recette, jamais des lignes de commandes de remplissage.

Le jeu conserve tous les identifiants, horaires, canaux et quantités de tickets de la simulation de 18 mois. Pour inclure le Double Steakhouse demandé dans les exemples, 40 % environ des unités commerciales Double Whopper/Menu Double Whopper ont été renommées par tirage reproductible au niveau des lignes en Double Steakhouse/Menu Double Steakhouse. Dans cette version fictive, ces deux doubles utilisent les mêmes deux pièces Whopper : les besoins de bœuf par ticket restent donc strictement inchangés. Le compteur exact est dans verification.json. Aucun prix n'est fourni dans cet export simplifié.

**Toutes les commandes et compositions sont SYNTHÉTIQUES.** Elles ne sont pas des recettes, prix ou données officiels de Burger King. La variante Double Steakhouse doit être ajoutée à la nomenclature d'un moteur existant avant ingestion, avec les identifiants P11 et M11.

CSV UTF-8, séparateur virgule, en-têtes en première ligne. produits_json est une liste JSON standard ; la colonne commande est destinée à la lecture humaine. Le ZIP contient les deux CSV principaux, l'extrait et les explications.
