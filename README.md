# Live Intelligence - API Custom

Un wrapper d'API basé sur Paradigm (LightOn) utile à des fins de divers projets internalisés. Adaptable aux produits utilisant la solution de LightOn pour l'IA générative.

Une série de tests unitaires est également mise à disposition dans le répertoire approprié.

## Description des classes

### 1. `ChatCompletions`

Génère des réponses à partir d'un modèle de langage dans un format de conversation (chat).

- **Méthodes principales :**
  - `execute(messages, model, history, **kwargs)`: Envoie une requête de complétion de chat.
  - `stream(messages, model, history, **kwargs)`: Diffuse la réponse de complétion de chat en continu.
- **Endpoint :** `POST /api/v2/chat/completions`

### 2. `Completions`

Génère des complétions de texte à partir d'un prompt.

- **Méthodes principales :**
  - `execute(prompt, model, **kwargs)`: Envoie une requête de complétion de texte.
  - `stream(prompt, model, **kwargs)`: Diffuse la réponse de complétion de texte en continu.
- **Endpoint :** `POST /api/v2/completions`

### 3. `DocumentSearch`

Effectue des recherches sémantiques dans des documents et génère des réponses basées sur les résultats.

- **Méthodes principales :**
  - `execute(query, model, workspace_ids, file_ids, **kwargs)`: Lance une recherche documentaire.
- **Endpoint :** `POST /api/v2/chat/document-search`

### 4. `AsyncDocumentSearch`

Version asynchrone de `DocumentSearch` pour exécuter des recherches documentaires de manière non bloquante, avec gestion de la concurrence.

- **Méthodes principales :**
  - `execute_single(session, query, **kwargs)`: Exécute une seule recherche documentaire de manière asynchrone.
  - `execute_batch(requests_data, max_concurrent, progress_callback)`: Exécute plusieurs recherches documentaires en parallèle.
- **Endpoint :** `POST /api/v2/chat/document-search` (utilisé en interne par les méthodes asynchrones)

### 5. `DocumentSearchAlt`

Une approche alternative pour la recherche documentaire, décomposée en récupération de chunks pertinents puis génération de réponse. Utile si l'endpoint `document-search` standard n'est pas optimal.

- **Méthodes principales :**
  - `execute(query, model, **kwargs)`: Orchestre la recherche alternative.
- **Endpoints internes utilisés :**
  - `POST /api/v2/query` (pour récupérer les chunks)
  - `POST /api/v2/chat/completions` (pour générer la réponse)

### 6. `Models`

Récupère la liste des modèles de langage disponibles.

- **Méthodes principales :**
  - `execute()`: Récupère les noms techniques des modèles.
- **Endpoint :** `GET /api/v2/models`

### 7. `Files`

Liste les fichiers accessibles (ID et nom) en fonction de la portée spécifiée (entreprise, privé, espace de travail).

- **Méthodes principales :**
  - `execute(company_scope, private_scope, workspace_scope, page)`: Récupère la liste des fichiers.
- **Endpoint :** `GET /api/v2/files`

### 8. `FileUploader`

Gère le téléversement de fichiers, incluant la gestion des sessions d'upload et le traitement par lots.

- **Méthodes principales :**
  - `open_session(ingestion_pipeline)`: Ouvre une nouvelle session d'upload.
  - `upload_file_to_session(session_uuid, file_path, **kwargs)`: Téléverse un fichier dans une session existante.
  - `get_session_details(session_uuid)`: Récupère les détails d'une session d'upload.
  - `delete_session(session_uuid)`: Supprime une session d'upload et ses documents.
  - `deactivate_all_sessions()`: Désactive les dernières sessions utilisateur.
  - `upload_files_in_batches(file_paths, batch_size, **kwargs)`: Gère le téléversement de plusieurs fichiers par lots.
  - `upload_files_to_personal_space(file_paths, **kwargs)`: Simplifie le téléversement de fichiers vers l'espace personnel.
- **Endpoints :**
  - `POST /api/v2/upload-session` (pour `open_session`)
  - `POST /api/v2/upload-session/{session_uuid}` (pour `upload_file_to_session`)
  - `GET /api/v2/upload-session/{session_uuid}` (pour `get_session_details`)
  - `DELETE /api/v2/upload-session/{session_uuid}` (pour `delete_session`)
  - `POST /api/v2/upload-session/deactivate` (pour `deactivate_all_sessions`)

Toutes les classes héritent de `ParadigmAPI`, qui gère l'authentification et la communication de base avec l'API.
