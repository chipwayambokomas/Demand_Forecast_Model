import pandas as pd
import os
import numpy as np

# --- Configuration ---

EXCEL_FILE_PATH = "data/raw_incoming_orders_data.xlsx"  
OUTPUT_CSV_PATH = "data/incoming_orders_data.csv"
# Remove rows with missing FNSKU in the final output
REMOVE_ROWS_WITH_MISSING_FNSKU = True
# Column names as they appear in your Excel sheets (CASE-SENSITIVE)
FNSKU_COL = "FNSKU"
DELIVERY_WEEK_COL = "Expected Arrival" 
UNITS_COL = "Units"         

# This list defines the columns we want and their order in the output CSV
EXPECTED_COLUMNS = [FNSKU_COL, DELIVERY_WEEK_COL, UNITS_COL]
# --- End Configuration ---

def extract_data_from_excel(excel_path, output_csv_path, expected_cols_config):
    """
    Reads an Excel file, iterates over its sheets, extracts specified columns,
    and saves the combined data to a CSV file.

    Args:
        excel_path (str): Path to the input .xlsx file.
        output_csv_path (str): Path to save the output .csv file.
        expected_cols_config (list): A list of column names to extract.
    """
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at '{excel_path}'")
        return

    all_sheets_data = []

    try:
        xls = pd.ExcelFile(excel_path)
    except Exception as e:
        print(f"Error: Could not open or read Excel file '{excel_path}'. Reason: {e}")
        print("Please ensure the file is a valid .xlsx file and not corrupted.")
        print("You might also need to install 'openpyxl': pip install openpyxl")
        return

    sheet_names = xls.sheet_names
    if not sheet_names:
        print(f"Warning: No sheets found in '{excel_path}'.")
        return

    print(f"Found {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")

    for sheet_name in sheet_names:
        print(f"Processing sheet: '{sheet_name}'...")
        try:
            # Read the current sheet
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)

            # Create a new DataFrame that will only contain the expected columns.
            # If an expected column is not present in df_sheet, it will be created
            # and filled with NaN (Not a Number), which represents missing data.
            # If df_sheet has extra columns, they will be ignored.
            # This also ensures the columns are in the order of expected_cols_config.
            extracted_df_for_sheet = df_sheet.reindex(columns=expected_cols_config)
            
            # Report on which columns were found or missing in this specific sheet
            actual_cols_found = [col for col in expected_cols_config if col in df_sheet.columns]
            missing_cols_in_sheet = [col for col in expected_cols_config if col not in df_sheet.columns]

            if not actual_cols_found:
                print(f"  -> Warning: None of the expected columns ({', '.join(expected_cols_config)}) found in sheet '{sheet_name}'. Skipping this sheet's data contribution if it's empty, or it will be all NaNs.")
            elif missing_cols_in_sheet:
                 print(f"  -> Note: Sheet '{sheet_name}' is missing columns: {', '.join(missing_cols_in_sheet)}. These will be filled with NaN.")
            else:
                print(f"  -> Successfully found all expected columns.")


            # Only add to our list if the extracted dataframe is not entirely empty
            # (i.e., it had some rows, even if some columns were all NaN)
            if not extracted_df_for_sheet.empty:
                all_sheets_data.append(extracted_df_for_sheet)
            else:
                print(f"  -> Note: Sheet '{sheet_name}' resulted in an empty dataset for the expected columns (it might have been an empty sheet or had no relevant columns with data).")


        except Exception as e:
            print(f"  -> Error processing sheet '{sheet_name}': {e}. Skipping this sheet.")
            continue

    if not all_sheets_data:
        print("No data extracted from any sheet, or all relevant sheets were empty/problematic. Output CSV will not be created.")
        return

    # Concatenate all the DataFrames from each sheet into one big DataFrame
    combined_df = pd.concat(all_sheets_data, ignore_index=True)
    combined_df.dropna(how='all', subset=expected_cols_config, inplace=True)

    if REMOVE_ROWS_WITH_MISSING_FNSKU:
        before = len(combined_df)
        combined_df = combined_df[combined_df[FNSKU_COL].notna()]
        after = len(combined_df)
        print(f"Removed {before - after} row(s) with missing FNSKU.")

    # Round up the units column to the next whole number
    if UNITS_COL in combined_df.columns:
        combined_df[UNITS_COL] = np.ceil(pd.to_numeric(combined_df[UNITS_COL], errors='coerce'))

    if combined_df.empty:
        print("After processing, the combined DataFrame is empty. No CSV file will be generated.")
        return

    try:
        combined_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nSuccessfully extracted data and saved to '{output_csv_path}'")
        print(f"Total rows in output: {len(combined_df)}")
        print(f"Columns in output: {', '.join(combined_df.columns.tolist())}")
    except Exception as e:
        print(f"\nError: Could not save data to CSV '{output_csv_path}'. Reason: {e}")

# --- Main execution ---
if __name__ == "__main__":
    extract_data_from_excel(EXCEL_FILE_PATH, OUTPUT_CSV_PATH, EXPECTED_COLUMNS)