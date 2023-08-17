# HOW TO OPERATE - GENERATE TABLE SQL & YML PYTHON SCRIPTS

See `README.md` for instructions on how to set up the `fsc-utils` dbt package in another repo. 

Python scripts for automating `.sql` and `.yml` dbt model creation.

1. Usage requires a `dbt_profile` and python `virtual environment (venv)` be set up to access dbt profile credentials and connection to Snowflake.
2. After installing the applicable version of `fsc-utils` in the respective repository, use the following commands to run these scripts.
- `python dbt_packages/fsc_utils/macros/python/generate_tbl_sql.py --config_file data/my_config_file_path` (see below for additional parameters or use -h)
- `python dbt_packages/fsc_utils/macros/python/generate_tbl_yml.py` (see below for additional parameters or use -h)
3. The `config_file` is a required JSON file that serves as the source for table generation. Update this file dynamically with new blockchains, contracts and topic_0s. 
- Each config_file may or may not be specific to a curation primitive/subject. 
- Each `object` in the JSON config file represents one (1) model to be generated. 
- Models will only be generated for the `blockchain` that you are working in (e.g. the repo/database in your CWD)

## GENERATE_TABLE_SQL.PY

Generates SQL/YML files based on a configuration file and stores them in the specified directory.
Note: this script calls the YML generation script by default.

    Parameters:
    - config_file (str, required): Path to the JSON configuration file which contains details like blockchain, schema, 
                         protocol, contract address, topic, and whether to drop the existing SQL file.
    
    - target (str, optional): Target environment, used for determining the DBT profile and database connection details. Default = dev.
    
    - drop_all (bool, optional): If set to True, it will drop and replace all SQL/YML files, ignoring the individual 
                                 'drop' settings in the config file. Default is False.

    Returns:
    None. This function generates SQL/YML files and saves them to the specified directory.

## GENERATE_TABLE_YML.PY

Generates a DBT .yml test file for each .sql file in the given directory or for a specified SQL file.
Note: this script may be used independently of the SQL generation script.
    
    Parameters:
    - model_paths (list of str): The paths to the directories containing .sql files or the paths to specific .sql files.
                            If a directory is provided for each path, the function will generate .yml files for all .sql files in those directories.
                            If specific .sql file paths are provided, the function will only generate a .yml file for those files.
    
    - output_dir (str, optional): The directory where the .yml files will be saved. If not provided, the .yml files will
                                  be saved in the same directory as the input .sql files or alongside the specified .sql file.
                                  
    - specific_files (list of str, optional): A list of specific .sql filenames for which .yml files should be generated.
                                              This argument is useful when the function is called from another script and
                                              you only want to generate .yml files for specific .sql files in a directory.
                                              If not provided and a directory is given for model_path, .yml files will be generated
                                              for all .sql files in that directory.
    
    Returns:
    None. This function writes .yml files to the specified or default output directory.