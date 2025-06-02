import os
import importlib

print("--- SASPy Configuration Diagnostic Script ---")

home_dir = os.path.expanduser('~')
sascfg_path_home = os.path.join(home_dir, 'sascfg_personal.py')

print(f"INFO: Your home directory is: {home_dir}")
print(f"INFO: Checking for sascfg_personal.py at: {sascfg_path_home}")

if os.path.exists(sascfg_path_home):
    print("SUCCESS: Found sascfg_personal.py in your home directory!")
    print("--- Contents of sascfg_personal.py ---")
    try:
        with open(sascfg_path_home, 'r') as f:
            print(f.read())
        print("-------------------------------------")
        print("INFO: Please VERIFY that the 'saspath' inside this file points to your actual SAS executable.")
    except Exception as e:
        print(f"ERROR: Could not read sascfg_personal.py. Error: {e}")
else:
    print("ERROR: sascfg_personal.py NOT FOUND in your home directory.")
    print("ACTION: You need to create this file. You can copy sascfg_personal.py.sample from your project directory to your home directory, rename it to sascfg_personal.py, and then EDIT the 'saspath' line in it.")

print("\nINFO: Attempting to import saspy and check its known configurations...")
saspy = None
try:
    saspy = importlib.import_module('saspy')
    saspy_sascfg = importlib.import_module('saspy.sascfg')
    print("SUCCESS: saspy and saspy.sascfg modules imported.")
    known_configs = getattr(saspy_sascfg, 'SAS_config_names', None)
    if known_configs:
        print(f"INFO: SAS_config_names found by saspy: {known_configs}")
        if not known_configs: # an empty list is possible if the file is there but SAS_config_names is empty
             print("WARNING: SAS_config_names is empty in the loaded config. Ensure it's defined like: SAS_config_names = ['default']")
    else:
        print("WARNING: saspy.sascfg was loaded, but it does not seem to have SAS_config_names defined or it is empty.")
        print("         This means saspy did not successfully parse config names from your sascfg_personal.py.")
        print("         Check for syntax errors in sascfg_personal.py or ensure SAS_config_names is defined (e.g., SAS_config_names = ['default']).")

except ImportError as e_imp:
    print(f"CRITICAL ERROR: Failed to import saspy or saspy.sascfg: {e_imp}")
    print("ACTION: Ensure saspy is installed in your Python environment (pip install saspy).")
    exit()
except Exception as e_cfg_inspect:
    print(f"ERROR during saspy config inspection: {e_cfg_inspect}")

print("\nINFO: Attempting to initialize SAS session (saspy.SASsession())...")
sas_session = None
try:
    if not saspy: # Should not happen if previous import worked
        print("ERROR: saspy module not available for session initialization.")
        exit()
    
    sas_session = saspy.SASsession() # Tries to use 'default' or first config
    print("\nSUCCESS: saspy.SASsession() object created!")
    
    print("INFO: Submitting a simple SAS command (PROC OPTIONS OPTION=VERSION)...")
    result = sas_session.submit("PROC OPTIONS OPTION=VERSION; RUN;")
    
    print("\n--- SAS Log ---")
    print(result.get('LOG', 'Log not available'))
    print("---------------")
    print("\n--- SAS Listing (LST) ---")
    print(result.get('LST', 'Listing not available or empty'))
    print("-------------------------")

except saspy.sasexceptions.SASConfigNotFoundError as e_cfg_err:
    print(f"\nCRITICAL SASPY CONFIGURATION ERROR: {e_cfg_err}")
    print("MESSAGE: saspy could not find any valid configuration to use.")
    print("ACTIONS:")
    print(f"  1. Ensure 'sascfg_personal.py' exists in your home directory: {sascfg_path_home}")
    print("  2. Ensure it defines 'SAS_config_names' (e.g., SAS_config_names = ['default']) and a corresponding configuration dictionary (e.g., default = {{...}}).")
    print("  3. Check for Python syntax errors within your 'sascfg_personal.py'.")
except saspy.sasexceptions.SASIOConnectionError as e_conn_err: # Catches SASIOConnectionTerminated
    print(f"\nSAS CONNECTION/IO ERROR: {e_conn_err}")
    print("MESSAGE: saspy found a configuration and tried to start SAS, but SAS failed to start/crashed or the connection failed.")
    print("ACTIONS:")
    print("  1. VERIFY the 'saspath' in your 'sascfg_personal.py' is the EXACT, CORRECT path to your SAS executable.")
    print("  2. Ensure your SAS software is correctly installed, licensed, and can run independently of Python.")
    print("  3. Check for issues with SAS itself (e.g., SAS logs if you can find them, permissions, SAS environment variables).")
except FileNotFoundError as e_fnf_err:
    print(f"\nFILE NOT FOUND ERROR during SAS startup: {e_fnf_err}")
    print("MESSAGE: The system could not find a file specified, likely the 'saspath' in your 'sascfg_personal.py'.")
    print("ACTION: Double-check the 'saspath' in 'sascfg_personal.py'. Make sure it is an exact path to an existing executable file.")
except PermissionError as e_perm_err:
    print(f"\nPERMISSION ERROR during SAS startup: {e_perm_err}")
    print("MESSAGE: The system denied permission to execute the 'saspath' or access related files.")
    print("ACTION: Check file permissions for your SAS installation and the 'saspath' executable. Ensure the user running Python has execute permissions.")
except Exception as e_general_err:
    print(f"\nAN UNEXPECTED GENERAL ERROR OCCURRED: {e_general_err}")
    print("This could be a variety of issues. The traceback below might provide more clues.")
    import traceback
    traceback.print_exc()
finally:
    if sas_session:
        print("\nINFO: Closing SAS session...")
        sas_session.endsas()
        print("INFO: SAS session closed.")
    print("\n--- Diagnostic Script Finished ---") 