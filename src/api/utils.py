import os
from fastapi import HTTPException

def validate_file_path(file_path: str, file_description: str):
    """
    Valide si un fichier existe et soulève une exception HTTP si ce n'est pas le cas.

    Args:
        file_path (str): Chemin vers le fichier.
        file_description (str): Description du fichier pour le message d'erreur.

    Raises:
        HTTPException: Si le fichier n'existe pas.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{file_description} non trouvé: {file_path}")