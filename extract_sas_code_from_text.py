import os

SOURCE_DIR = "SAS_extracted_text/"
DEST_DIR = "SAS_scripts_auto_extracted/"
MARKER = "/*  Pas"

def extract_sas_snippets():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory '{SOURCE_DIR}' not found.")
        return

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory '{DEST_DIR}'.")

    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".txt"):
            source_filepath = os.path.join(SOURCE_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                with open(source_filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {source_filepath}: {e}")
                continue

            snippet_count = 0
            in_snippet = False
            current_snippet_lines = []

            for line in lines:
                if MARKER in line:
                    if in_snippet and current_snippet_lines: # Save previous snippet
                        snippet_count += 1
                        snippet_filename = f"{base_name}_snippet_{snippet_count}.sas"
                        dest_filepath = os.path.join(DEST_DIR, snippet_filename)
                        try:
                            with open(dest_filepath, 'w', encoding='utf-8') as sf:
                                sf.writelines(current_snippet_lines)
                            print(f"Saved: {dest_filepath}")
                        except Exception as e:
                            print(f"Error writing {dest_filepath}: {e}")
                    
                    # Start new snippet (lines after the marker line)
                    in_snippet = True
                    current_snippet_lines = [] 
                    # We don't include the marker line itself, but what's *under* it.
                    # If code starts on the *same* line after marker, this needs adjustment.
                    # Assuming code starts on the NEXT line.
                elif in_snippet:
                    current_snippet_lines.append(line)
            
            # Save the last snippet if any
            if in_snippet and current_snippet_lines:
                snippet_count += 1
                snippet_filename = f"{base_name}_snippet_{snippet_count}.sas"
                dest_filepath = os.path.join(DEST_DIR, snippet_filename)
                try:
                    with open(dest_filepath, 'w', encoding='utf-8') as sf:
                        sf.writelines(current_snippet_lines)
                    print(f"Saved: {dest_filepath}")
                except Exception as e:
                    print(f"Error writing {dest_filepath}: {e}")
            
            if snippet_count == 0:
                print(f"No snippets found (marker '{MARKER}') in {filename}.")

if __name__ == "__main__":
    extract_sas_snippets()
    print("\nReminder: Please MANUALLY REVIEW all generated .sas files in")
    print(f"'{DEST_DIR}' for correctness and completeness before use.")
    print("You still need to configure saspy (sascfg_personal.py) for your environment.") 