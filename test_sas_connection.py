import saspy
import os

print("--- SASPY Connection Test ---")
print(f"Current working directory: {os.getcwd()}")
print(f"User home directory: {os.path.expanduser('~')}")

# Attempt to print the location of sascfg_personal.py if saspy can find its config module
try:
    import saspy.sascfg
    # Note: saspy doesn't directly expose the path of the loaded cfg file easily.
    # It searches predefined locations.
    print("saspy.sascfg module imported. It will search for sascfg.py or sascfg_personal.py.")
    print(f"Default config names saspy knows: {saspy.SAScfg.SAS_config_names if hasattr(saspy.SAScfg, 'SAS_config_names') else 'Not found'}")
except ImportError:
    print("ERROR: Could not import saspy.sascfg. This is unexpected if saspy is installed.")
except Exception as e_cfg_load:
    print(f"ERROR trying to access saspy.sascfg: {e_cfg_load}")

print("\nAttempting to initialize SAS session...")
sas_session = None
try:
    # If you named your config something other than 'default' in sascfg_personal.py,
    # specify it here, e.g., saspy.SASsession(cfgname='yourconfigname')
    sas_session = saspy.SASsession()
    print("\nSUCCESS: SAS session object created!")
    
    print("Submitting a simple SAS command (PROC OPTIONS)...")
    # Example: Get the SAS version
    result = sas_session.submit("PROC OPTIONS OPTION=VERSION; RUN;")
    
    print("\nSAS Log:")
    print(result.get('LOG', 'Log not available')) # saspy might return log/lst as dict keys

    print("\nSAS Listing (LST):")
    print(result.get('LST', 'Listing not available or empty'))

except saspy.sasexceptions.SASConfigNotFoundError as e_cfg:
    print(f"\nCRITICAL CONFIGURATION ERROR: {e_cfg}")
    print("saspy could not find its configuration file (sascfg.py or sascfg_personal.py) or a valid configuration entry.")
    print(f"ACTION: Ensure 'sascfg_personal.py' exists in your home directory ('{os.path.expanduser('~')}') or another location in Python's sys.path, and that it contains a valid configuration.")
except saspy.sasexceptions.SASIOConnectionError as e_conn: # Catches SASIOConnectionTerminated
    print(f"\nSAS CONNECTION/IO ERROR: {e_conn}")
    print("This usually means saspy found a configuration and tried to start SAS, but SAS failed to start or crashed.")
    print("ACTIONS:")
    print("1. Verify the 'saspath' in your 'sascfg_personal.py' is the EXACT, CORRECT path to your SAS executable.")
    print("2. Ensure your SAS software is correctly installed, licensed, and can run independently.")
    print("3. Check for issues with SAS itself (e.g., SAS logs if you can find them, permissions).")
except FileNotFoundError as e_fnf:
    print(f"\nFILE NOT FOUND ERROR during SAS startup: {e_fnf}")
    print("This often means the 'saspath' in your 'sascfg_personal.py' points to a file/directory that doesn't exist.")
    print("ACTION: Double-check the 'saspath' in 'sascfg_personal.py'.")
except PermissionError as e_perm:
    print(f"\nPERMISSION ERROR during SAS startup: {e_perm}")
    print("The system denied permission to execute the 'saspath' or access related files.")
    print("ACTION: Check file permissions for your SAS installation and the 'saspath' executable.")
except Exception as e_general:
    print(f"\nAN UNEXPECTED ERROR OCCURRED: {e_general}")
    print("This could be a variety of issues. The traceback might provide more clues.")
    import traceback
    traceback.print_exc()
finally:
    if sas_session:
        print("\nClosing SAS session...")
        sas_session.endsas()
        print("SAS session closed.")
    print("\n--- Test Finished ---") 