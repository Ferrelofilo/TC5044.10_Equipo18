# Agregar en este nivel archivo flare-mlops###.json
# Para autorización de cuenta servicio de google drive.
## Para inicializar se puede correr el siguiente comando en terminal:
*dvc remote modify remote-storage --local  gdrive_service_account_json_file_path {root content .json file path}*
## reemplaza { } por el path adecuado.
Dentro del archivo config.local terminaría viendose como algo:

['remote "remote-storage"']

gdrive_service_account_json_file_path = <nombre completo flare-mlops###.json>