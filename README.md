# Projet N11 — Génération de QCM avec validation automatique

Le projet contient quatre notebooks exécutables dans l'ordre :

1. [notebooks/01_preparation_donnees.ipynb](notebooks/01_preparation_donnees.ipynb)
2. [notebooks/02_generation_qcm.ipynb](notebooks/02_generation_qcm.ipynb)
3. [notebooks/03_validation_automatique.ipynb](notebooks/03_validation_automatique.ipynb)
4. [notebooks/04_evaluation_experimentale.ipynb](notebooks/04_evaluation_experimentale.ipynb)

Les six supports fournis sont copiés dans `data/sources/`. L'OCR des pages
scannées utilise RapidOCR. La baseline de validation reste le QA local étudié
(TF-IDF et recouvrement lexical). La méthode avancée ajoute le validateur QA
Gemini en français. Seuls les extraits de cours, les questions et les options
soumis au validateur Gemini sont envoyés à Google ; l'appel utilise
`store=False`. L'API nécessite Internet et reste soumise aux quotas du compte.

## Installation et exécution

Dans PowerShell, depuis la racine du projet :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name n11-qcm --display-name "Python (N11 QCM)"
jupyter notebook
```

Créer une clé Gemini dans [Google AI Studio](https://aistudio.google.com/apikey),
ouvrir le fichier `.env` à la racine du projet (ou copier `.env.example` si le
fichier est absent) et renseigner `GEMINI_API_KEY`. Le fichier `.env` est ignoré
par Git ; ne pas le joindre à l'article, à l'archive de remise ou à un partage.
Ne jamais coller la clé dans un notebook ni dans le code. Le modèle par défaut
est `gemini-3.5-flash-lite` ;
`GEMINI_MODEL` dans `.env` permet de choisir un autre modèle autorisé par le
compte. Une variable d'environnement Windows peut aussi être utilisée à la
place du fichier local.

Sélectionner le noyau `Python (N11 QCM)`, puis exécuter les quatre notebooks de
haut en bas. Le notebook 02 produit la baseline TF-IDF ; le notebook 03 compare
le filtre local et Gemini sur les mêmes candidats. Les résultats Gemini sont
distincts des fichiers locaux dans `data/processed/`. Les artefacts sont écrits
dans `data/processed/` et `results/`.

## Portée de cette première version

La génération de référence est extractive et déterministe : elle construit les
questions à partir de phrases des supports et choisit des distracteurs issus
des mêmes documents. TF-IDF est conservé comme baseline étudiée. Pour la
validation avancée, Gemini évalue si la question est répondable à partir du
contexte, sélectionne l'option et doit fournir une citation exacte vérifiable
localement. Cette sortie d'API n'est pas une confiance calibrée ni une
validation humaine ; toute question doit être relue.

La grille complétée par deux évaluatrices humaines est enregistrée dans
`data/processed/grille_evaluation_humaine.csv` : elle comprend 60 QCM, dont 53
retenus et 7 rejetés par le validateur local TF-IDF. Ces annotations portent sur
ces QCM et ne doivent pas être présentées comme une évaluation humaine des
résultats Gemini. Le notebook 04 vérifie l'alignement des
identifiants, questions, décisions QA et notes (échelle 1–5), puis calcule les
kappas, les moyennes par décision QA et les résultats comparatifs dans
`results/evaluation_humaine_metrics.csv`,
`results/evaluation_humaine_comparative.csv` et
`results/evaluation_humaine_resume.md`. Il régénère aussi `article_projet.docx`,
dont la pagination doit être vérifiée dans Word ou LibreOffice avant la remise.
Le fichier de synthèse reçu avec les notes mentionnait un LLM ; le groupe a
confirmé que les notes sont celles de deux évaluatrices humaines, provenance
retenue dans le projet.
