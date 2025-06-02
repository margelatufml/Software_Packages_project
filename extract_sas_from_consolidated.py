import os

SOURCE_FILE = "all_sas_documents_content.txt"
DEST_FILE = "code.txt"
MARKER = "/*  Pas"

def extract_sas_from_consolidated_file():
    if not os.path.exists(SOURCE_FILE):
        print(f"Source file '{SOURCE_FILE}' not found.")
        return

    all_extracted_code = []
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {SOURCE_FILE}: {e}")
        return

    in_snippet = False
    current_snippet_lines = []

    for line in lines:
        if MARKER in line:
            if in_snippet and current_snippet_lines: # Append previous snippet to all_extracted_code
                all_extracted_code.extend(current_snippet_lines)
                all_extracted_code.append("\n" + "-" * 20 + " End of Snippet " + "-" * 20 + "\n\n") # Add a separator
            
            # Start new snippet (lines after the marker line)
            in_snippet = True
            current_snippet_lines = [] 
            # We don't include the marker line itself, but what's *under* it.
            # Assuming code starts on the NEXT line.
        elif in_snippet:
            current_snippet_lines.append(line)
    
    # Append the last snippet if any
    if in_snippet and current_snippet_lines:
        all_extracted_code.extend(current_snippet_lines)

    if not all_extracted_code:
        print(f"No code snippets found (using marker '{MARKER}') in {SOURCE_FILE}.")
        # Create an empty code.txt if no snippets are found, as per user request for the file to exist
        try:
            with open(DEST_FILE, 'w', encoding='utf-8') as df:
                df.write("# No SAS code snippets found based on the marker.\n")
            print(f"Created empty '{DEST_FILE}' as no snippets were found.")
        except Exception as e:
            print(f"Error writing empty {DEST_FILE}: {e}")
        return

    try:
        with open(DEST_FILE, 'w', encoding='utf-8') as df:
            df.writelines(all_extracted_code)
        print(f"All extracted SAS code snippets saved to '{DEST_FILE}'.")
    except Exception as e:
        print(f"Error writing {DEST_FILE}: {e}")

if __name__ == "__main__":
    extract_sas_from_consolidated_file()
    print("\nReminder: Please MANUALLY REVIEW the content of code.txt.")
    print("This extraction is based on a heuristic and may not be perfect.")
    print("You still need to configure saspy (sascfg_personal.py) for your environment.") 